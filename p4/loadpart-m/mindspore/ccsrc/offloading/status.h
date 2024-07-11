#ifndef MINDSPORE_OFFLOADING_STATUS_H
#define MINDSPORE_OFFLOADING_STATUS_H
namespace mindspore {
namespace offloading {
enum Status {
  SUCCESS = 0,
  FAILED,
  INVALID_ARGUMENT,
};
}  // namespace offloading
}  // namespace mindspore

#endif  // MINDSPORE_OFFLOADING_STATUS_H