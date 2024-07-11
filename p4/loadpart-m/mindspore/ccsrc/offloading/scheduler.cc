#include "scheduler.h"

namespace mindspore {

namespace offloading {

void SeparateBatchTensor(TensorPtr& tensor, std::vector<offloading_serving::PredictReply*>& replies) {
  MS_EXCEPTION_IF_NULL(tensor);
  // Assume the first dim is always batch dim
  const auto &dims = tensor->shape();
  size_t batch_size = dims[0];
  if (batch_size != replies.size()) {
    MS_LOG(EXCEPTION) << "Batch size and the number of replies are not equal";
    return;
  }
  auto single_batch_size = tensor->data().nbytes() / batch_size;
  auto dtype = GetProtoDataType(tensor->data_type());
  auto tensor_data_c = reinterpret_cast<uint8_t *>(tensor->data_c());
  MS_EXCEPTION_IF_NULL(tensor_data_c);
  for (size_t i = 0; i < batch_size; ++i) {
    auto& reply = replies[i];
    MS_EXCEPTION_IF_NULL(reply);
    offloading_serving::TensorProto *tensor_proto = reply->add_tensor();
    // Separate batch & write proto
    tensor_proto->set_data_type(dtype);
    tensor_proto->add_dims(1);
    for (size_t i = 1; i < dims.size(); ++i) {
      tensor_proto->add_dims(dims[i]);
    }
    tensor_proto->set_name(tensor->id());
    tensor_proto->set_raw_data(tensor_data_c, single_batch_size);
    tensor_proto->set_name("");
    tensor_proto->set_compressed(false);

    tensor_data_c += single_batch_size;
  }
}

void Scheduler::Init(OffloadingServerConfig &cfg, const std::string &path) {
  auto prep_exec = std::make_shared<PreprocessExecutor>(cfg.decompress_nthreads, cfg.is_dequant);
  prep_exec->Init();
  stages_.emplace_back(prep_exec);

  auto inf_exec = std::make_shared<InferenceExecutor>(cfg.is_dequant);
  inf_exec->Init(path, cfg.dequant_path, cfg.max_batch_size, cfg.prep_full_graphs);
  stages_.emplace_back(inf_exec);

  for (size_t i = 0; i < stages_.size(); ++i) {
    stages_[i]->SetScheduler(shared_from_this());
    if (i != stages_.size() - 1) {
      stages_[i]->SetNext(stages_[i + 1]);
    } else {
      stages_[i]->SetLastStage(true);
    }
  }
  
  if (!is_sched_running_.load()) {
    is_sched_running_.store(true);
  }
  sched_thread_ = std::thread(&Scheduler::Run, this);
  if (!is_completion_running_.load()) {
    is_completion_running_.store(true);
  }
  completion_thread_ = std::thread(&Scheduler::ProcessResponse, this);

  if (cfg.time_log) {
    time_log_ofs.open("time.log");
    if (!time_log_ofs.is_open()) {
      MS_LOG(EXCEPTION) << "Scheduler: open time.log failed";
    }
  }
}

void Scheduler::Stop() {
  // stop executors
  for (auto& s : stages_) {
    s->Stop();
  }
  if (is_sched_running_.load()) {
    is_sched_running_.store(false);
    sched_thread_.join();
  }
  if (is_completion_running_.load()) {
    is_completion_running_.store(false);
    completion_thread_.join();
  }

  if (time_log_ofs.is_open()) {
    time_log_ofs.close();
  }
}

void Scheduler::ProcessRequest(offloading_serving::PredictRequest* request,
                    offloading_serving::PredictReply* response,
                    PredictOnFinish& callback) {
  // register InferenceSession
  auto id = GetNextID();
  auto session = std::make_shared<InferenceSession>(callback, response);
  {
    std::lock_guard<std::mutex> lk(inf_session_map_lock_);
    inf_session_map_[id] = session;
  }
  // Enqueue
  PayloadPtr payload = std::make_shared<Payload>(std::initializer_list<offloading_serving::PredictRequest*>{request}, std::initializer_list<uint64_t>{id});
  payload->q_time_ = TIMESTAMP();
  payload_queue_.Enqueue(payload);
}

void Scheduler::EnqueueCompletionQueue(TaskPtr&& task) {
  completion_queue_.PushTask(std::move(task));
}

void Scheduler::GetRuntimeProfile(double &factor, double &q_time) {
  mtx_.lock();
  factor = k_buf_.GetAvgValue();
  q_time = q_time_buf_.GetAvgValue();
  mtx_.unlock();
}

void Scheduler::Run() {
  while (is_sched_running_.load()) {
    switch (mode_) {
      case NO_BATCHING:
        ScheduleNoBatching();
        break;
      case NAIVE_BATCHING:
        ScheduleNaiveBatching();
        break;
      case SNB_BATCHING:
        ScheduleSNBBatching();
        break;
      default:
        ScheduleNoBatching();
        break;
    }
  }
}

void Scheduler::ScheduleNoBatching() {
  auto next_payload = payload_queue_.Dequeue();

  auto dequeue_time = TIMESTAMP();
  next_payload->q_time_ = dequeue_time - next_payload->q_time_;

  MS_EXCEPTION_IF_NULL(next_payload);
  auto plan = std::make_shared<ExecutionPlan>();
  plan->bsz_list.push_back(1);
  plan->entry_list.push_back(std::make_pair(next_payload->requests_[0]->lg_idx(), next_payload->requests_[0]->cut_point()));
  plan->mode = NO_BATCHING;
  next_payload->plan_ = plan;
  
  auto first_stage = std::dynamic_pointer_cast<PreprocessExecutor>(stages_[0]);
  if (!first_stage) {
    MS_LOG(EXCEPTION) << "Server: The first StageExecutor should be PreprocessExecutor";
    return;
  }

  auto sched_finish_time = TIMESTAMP();
  next_payload->e_time_ = sched_finish_time - dequeue_time;
  next_payload->req_q_times_.emplace_back(next_payload->q_time_);
  next_payload->q_time_ = sched_finish_time;

  first_stage->EnqueuePayload(next_payload);
}

void Scheduler::ScheduleNaiveBatching() {
  std::vector<PayloadPtr> payloads;
  // always assume each payload here contains one request
  payload_queue_.DequeueBatch(payloads);

  auto dequeue_time = TIMESTAMP();

  if (payloads.empty()) return;
  // sort & batch payloads in different sizes
  std::sort(payloads.begin(), payloads.end(), [](PayloadPtr &lhs, PayloadPtr &rhs) {
    auto lhs_request = lhs->requests_[0];
    auto rhs_request = rhs->requests_[0];
    return (lhs_request->lg_idx() < rhs_request->lg_idx()) 
            || (lhs_request->lg_idx() == rhs_request->lg_idx() && lhs_request->cut_point() < rhs_request->cut_point());
  });
  
  auto first_stage = std::dynamic_pointer_cast<PreprocessExecutor>(stages_[0]);
  if (!first_stage) {
    MS_LOG(EXCEPTION) << "Server: The first StageExecutor should be PreprocessExecutor";
    return;
  }

  size_t cur_lg_idx = payloads[0]->requests_[0]->lg_idx();
  auto cur_cut_point = payloads[0]->requests_[0]->mutable_cut_point();
  auto next_payload = std::make_shared<Payload>();
  for (auto &payload : payloads) {
    auto &req = payload->requests_[0];
    if ((size_t)req->lg_idx() == cur_lg_idx) {
      if (req->cut_point() != *cur_cut_point) {
        MS_LOG(EXCEPTION) << "Scheduler: find different cut points inside the same latency graph group, not supported";
      }
    } else {
      auto plan = std::make_shared<ExecutionPlan>();
      plan->bsz_list.push_back(next_payload->requests_.size());
      plan->entry_list.push_back(std::make_pair(cur_lg_idx, *cur_cut_point));
      plan->mode = NAIVE_BATCHING;
      next_payload->plan_ = plan;
      first_stage->EnqueuePayload(next_payload);

      cur_lg_idx = req->lg_idx();
      cur_cut_point = req->mutable_cut_point();
      next_payload = std::make_shared<Payload>();
    }
    next_payload->requests_.emplace_back(req);
    next_payload->session_ids_.emplace_back(payload->session_ids_[0]);
  }
  auto plan = std::make_shared<ExecutionPlan>();
  plan->bsz_list.push_back(next_payload->requests_.size());
  plan->entry_list.push_back(std::make_pair(cur_lg_idx, *cur_cut_point));
  plan->mode = NAIVE_BATCHING;
  next_payload->plan_ = plan;

  for (auto &p : payloads) {
    next_payload->req_q_times_.emplace_back(dequeue_time - p->q_time_);
  }
  auto sched_finish_time = TIMESTAMP();
  next_payload->e_time_ = sched_finish_time - dequeue_time;
  next_payload->q_time_ = sched_finish_time;

  first_stage->EnqueuePayload(next_payload);
}

void Scheduler::ScheduleSNBBatching() {
  std::vector<PayloadPtr> payloads;
  // always assume each payload here contains one request
  payload_queue_.DequeueBatch(payloads);

  auto dequeue_time = TIMESTAMP();

  if (payloads.empty()) return;
  // sort & batch payloads in different sizes
  std::sort(payloads.begin(), payloads.end(), [](PayloadPtr &lhs, PayloadPtr &rhs) {
    auto lhs_request = lhs->requests_[0];
    auto rhs_request = rhs->requests_[0];
    return (lhs_request->lg_idx() < rhs_request->lg_idx()) 
            || (lhs_request->lg_idx() == rhs_request->lg_idx() && lhs_request->cut_point() < rhs_request->cut_point());
  });  

  size_t cur_bsz = 0;
  size_t cur_lg_idx = payloads[0]->requests_[0]->lg_idx();
  auto cur_cut_point = payloads[0]->requests_[0]->mutable_cut_point();
  auto next_payload = std::make_shared<Payload>();
  auto plan = std::make_shared<ExecutionPlan>();
  for (auto &payload : payloads) {
    auto &req = payload->requests_[0];
    if ((size_t)req->lg_idx() == cur_lg_idx) {
      if (req->cut_point() != *cur_cut_point) {
        MS_LOG(EXCEPTION) << "Scheduler: find different cut points inside the same latency graph group, not supported";
      }
    } else {
      plan->bsz_list.push_back(cur_bsz);
      plan->entry_list.push_back(std::make_pair(cur_lg_idx, *cur_cut_point));
      cur_bsz = 0;
      cur_lg_idx = req->lg_idx();
      cur_cut_point = req->mutable_cut_point();
    }
    cur_bsz++;
    next_payload->requests_.emplace_back(req);
    next_payload->session_ids_.emplace_back(payload->session_ids_[0]);
  }
  plan->bsz_list.push_back(cur_bsz);
  plan->entry_list.push_back(std::make_pair(cur_lg_idx, *cur_cut_point));
  plan->mode = SNB_BATCHING;
  next_payload->plan_ = plan;
  
  auto first_stage = std::dynamic_pointer_cast<PreprocessExecutor>(stages_[0]);
  if (!first_stage) {
    MS_LOG(EXCEPTION) << "Server: The first StageExecutor should be PreprocessExecutor";
    return;
  }

  for (auto &p : payloads) {
    next_payload->req_q_times_.emplace_back(dequeue_time - p->q_time_);
  }
  auto sched_finish_time = TIMESTAMP();
  next_payload->e_time_ = sched_finish_time - dequeue_time;
  next_payload->q_time_ = sched_finish_time;

  first_stage->EnqueuePayload(next_payload);
}

void Scheduler::ProcessResponse() {
  while (is_completion_running_.load()) {
    auto task = completion_queue_.PopTask();

    auto dequeue_time = TIMESTAMP();

    std::vector<InferenceSessionPtr> sessions;
    {
      std::lock_guard<std::mutex> lk(inf_session_map_lock_);
      for (auto& id : task->session_ids) {
        auto it = inf_session_map_.find(id);
        if (it == inf_session_map_.end()) {
          MS_LOG(EXCEPTION) << "Server: unknown InferSession ID: " << id;
          return;
        }
        sessions.emplace_back(it->second);
        inf_session_map_.erase(id);
      }
    }
    std::vector<offloading_serving::PredictReply*> replies;
    std::vector<PredictOnFinish> ret_cbs;
    for (auto& s : sessions) {
      replies.emplace_back(s->reply);
      ret_cbs.emplace_back(s->callback);
    }
    if (task->inputs.size() != 1 && task->inputs[0].size() != 1) {
      MS_LOG(EXCEPTION) << "Scheduler: the completion thread can only process Task with 1 tensor";
      return;
    }
    // convert to ProtoTensor, maybe slice Tensor
    auto ret_tensor = task->inputs[0].begin()->second.front();
    SeparateBatchTensor(ret_tensor, replies);
    
    task->q_times[2] = dequeue_time - task->q_times[2];
    auto exec_time = TIMESTAMP() - dequeue_time;
    
    int64_t total_queue_time = task->q_times[0] + task->q_times[1] + task->q_times[2];
    int64_t overhead = task->e_times[0] + task->e_times[1] + exec_time;
    double avg_first_q_time = (double)std::accumulate(task->req_q_times.begin(), task->req_q_times.end(), 0) / (double)task->req_q_times.size();
    double q_time = (double)total_queue_time + avg_first_q_time + (double)overhead;

    if (time_log_ofs.is_open()) {
      time_log_ofs << replies.size() << "\t\t"
                   << task->e_times[0] << "\t\t" 
                   << task->q_times[0] << "\t\t"
                   << task->e_times[1] << "\t\t"
                   << task->q_times[1] << "\t\t"
                   << task->e_times[2] << "\t\t"
                   << task->q_times[2] << "\t\t"
                   << exec_time << "\t\t"
                   << task->factor << "\t\t"
                   << q_time << std::endl;
    }
    // write to load buffer
    mtx_.lock();
    k_buf_.Push(task->factor);
    q_time_buf_.Push(q_time);
    mtx_.unlock();

    // call callbacks correctly 
    for (size_t i = 0; i < replies.size(); ++i) {
      auto& reply = replies[i];  
      auto& cb_func = ret_cbs[i];

      double cur_q_time = total_queue_time + task->req_q_times[i] + overhead;
      reply->set_q_time(cur_q_time);
      reply->set_factor(task->factor);

      cb_func(reply, grpc::Status::OK);
    }
    
  }
}

uint64_t Scheduler::GetNextID() {
  static std::atomic<uint64_t> id = 0;
  return ++id;
}

}
}