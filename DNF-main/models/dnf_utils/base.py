import mindspore
from mindspore import nn , ops

from .fuse import PDConvFuse
from .cid import CID
from .mcc import MCC
from .sample import SimpleUpsample, SimpleDownsample
from .resudual_switch import ResidualSwitchBlock

from abc import abstractmethod



class DNFBase(nn.Cell):
    def __init__(self, f_number, *,
                 block_size=1,
                 layers=4,
                 denoising_block='CID',
                 color_correction_block='MCC'
                 ) -> None:
        super().__init__()
        # Define a helper function to help get the class of a str or the class itself
        def get_class(class_or_class_str):
            return eval(class_or_class_str) if isinstance(class_or_class_str , str) else class_or_class_str

        # This way we can get all the needed classes
        self.denoising_block_class = get_class(denoising_block)
        self.color_correction_block_class = get_class(color_correction_block)
        self.downsample_class = SimpleDownsample
        self.upsample_class = SimpleUpsample
        self.decoder_fuse_class = PDConvFuse

        self.pad_mode = 'pad'
        self.act = nn.GELU()
        self.layers = layers

        # Used for color correction block
        head = [2 ** layer for layer in range(layers)]
        self.block_size = block_size
        inchannel = 3 if block_size == 1 else block_size * block_size
        outchannel = 3 * block_size * block_size

        self.feature_conv_0 = nn.Conv2d(inchannel, f_number, kernel_size=5, stride=1, padding=2, has_bias=True, pad_mode=self.pad_mode)
        self.feature_conv_1 = nn.Conv2d(f_number, f_number, kernel_size=5, stride=1, padding=2, has_bias=True, pad_mode=self.pad_mode)

        self.downsamples = nn.CellList([
            self.downsample_class(
                f_number * (2 ** idx),
                pad_mode=self.pad_mode
            )
            for idx in range(layers - 1)
        ])

        self.upsamples = nn.CellList([
            self.upsample_class(
                f_number * (2 ** idx),
                pad_mode=self.pad_mode
            )
            for idx in range(1 , layers)
        ])

        self.denoising_blocks = nn.CellList([
            ResidualSwitchBlock(
                self.denoising_block_class(
                    f_number=f_number * (2 ** idx),
                    pad_mode = self.pad_mode
                )
            )
            for idx in range(layers)
        ])

        self.denoising_blocks = nn.CellList([
            ResidualSwitchBlock(
                self.denoising_block_class(
                    f_number=f_number * (2 ** idx),
                    pad_mode = self.pad_mode
                )
            )
            for idx in range(layers)
        ])

        self.color_correction_blocks = nn.CellList([
            self.color_correction_block_class(
                f_number=f_number * (2 ** idx),
                num_heads=head[idx],
                pad_mode = self.pad_mode
            )
            for idx in range(layers)
        ])

        self.color_decoder_fuses = nn.CellList([
            self.decoder_fuse_class(
                in_channels=f_number*(2 ** idx)
            )
            for idx in range(layers - 1)
        ])

        self.conv_fuse_0 = nn.Conv2d(f_number,f_number,kernel_size=3,stride=1,padding=1,pad_mode='pad',has_bias=True)
        self.conv_fuse_1 = nn.Conv2d(f_number,outchannel,kernel_size=1,stride=1,padding=0,pad_mode='valid',has_bias=True)

        if block_size > 1:
            self.pixel_shuffle = nn.layer.image.PixelShuffle(block_size)
        else:
            self.pixel_shuffle = nn.Identity()

    @abstractmethod
    def _pass_features_to_color_decoder(self , x , f_short_cut , encoder_features):
        pass

    def _check_and_padding(self, x):
        # Calculate the required size based on the input  size and required factor
        _, _, h, w = x.size()
        stride = (2 ** (self.layers - 1))

        # Calculate the number of pixels needed to reach the required size
        dh = -h % stride
        dw = -w % stride

        # Calculate the amount of padding needed for each side
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w+left_pad, top_pad, h+top_pad)

        # Pad the tensor with reflect mode
        padded_tensor = ops.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect",value=0
        )

        return padded_tensor

    def _check_and_crop(self, x, res1):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top*self.block_size:bottom*self.block_size, left*self.block_size:right*self.block_size]
        res1 = res1[:, :, top:bottom, left:right] if res1 is not None else None
        return x, res1

    def construct(self, x):
        x = self._check_and_padding(x)
        x = self.act(self.feature_conv_0(x))
        x = self.feature_conv_1(x)
        f_short_cut = x

        ## encoder, local residual switch off
        encoder_features = []
        for denoise, down in zip(self.denoising_blocks[:-1], self.downsamples):
            x = denoise(x, 0)  # residual switch off
            encoder_features.append(x)
            x = down(x)
        x = self.denoising_blocks[-1](x, 0)  # residual switch off

        x, res1, refined_encoder_features = self._pass_features_to_color_decoder(x, f_short_cut, encoder_features)

        ## color correction
        for color_correction, up, fuse, encoder_feature in reversed(list(zip(
            self.color_correction_blocks[1:],
            self.upsamples,
            self.color_decoder_fuses,
            refined_encoder_features
        ))):
            x = color_correction(x)
            x = up(x)
            x = fuse(x, encoder_feature)
        x = self.color_correction_blocks[0](x)

        x = self.act(self.conv_fuse_0(x))
        x = self.conv_fuse_1(x)
        x = self.pixel_shuffle(x)
        rgb, raw = self._check_and_crop(x, res1)
        return rgb, raw