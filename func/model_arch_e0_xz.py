# model
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from functools import partial
from .Dsc_conv import DCN_Conv


# for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding, stride=stride, dilation=dilation):
def create_conv(
    in_channels,
    out_channels,
    kernel_size,
    order,
    num_groups,
    padding,
    stride,
    dilation=1,
):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    创建一个list，元素为单卷积模块(包含BN/GN)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    Return:
        list of tuple (name, module)
    """
    assert "c" in order, "Conv layer MUST be present"
    assert (
        order[0] not in "rle"
    ), "Non-linearity cannot be the first operation in the layer"

    modules = []
    # order=gcr,看看是不是全是gcr
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(
                ("LeakyReLU", nn.LeakyReLU(negative_slope=0.1, inplace=True))
            )
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "c":
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ("g" in order or "b" in order)
            modules.append(
                (
                    "conv",
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        padding=padding,
                        bias=bias,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
            )
        elif char == "g":
            is_before_conv = i < order.index("c")  # 在conv之前用g还是之后用
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            # 对于GN，如果通道维较少，则只能分成一个组
            if num_channels < num_groups:  # num_groups=8
                num_groups = 1

            assert (
                num_channels % num_groups == 0
            ), f"Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}"
            modules.append(
                (
                    "groupnorm",
                    nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
                )
            )
        elif char == "b":
            is_before_conv = i < order.index("c")
            if is_before_conv:
                modules.append(("batchnorm", nn.BatchNorm3d(in_channels)))
            else:
                modules.append(("batchnorm", nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(
                f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']"
            )

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    基础的卷积module，由conv3d、非线性的BN或GN
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    # self.conv1 = SingleConv(in_channels, middle_channels, conv_kernel_size, conv_layer_order, num_groups, padding=padding, stride=stride)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        order="gcr",
        num_groups=8,
        padding=1,
        stride=1,
        dilation=1,
    ):
        super(SingleConv, self).__init__()
        # creat_conv的output是一个字典
        for name, module in create_conv(
            in_channels,
            out_channels,
            kernel_size,
            order,
            num_groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
        ):
            self.add_module(name, module)


class AttModule(nn.Module):  # SE_Net
    def __init__(self, channel, mid_channel=8):
        super(AttModule, self).__init__()
        # nn.AdaptiveAvgPool3d我们只需要关注输出维度的大小 output_size ，具体的实现过程和参数选择自动帮你确定了。
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        # print(x.shape)#torch.Size([4, 32, 32, 128, 128])
        y = self.avg_pool(x).view(b, c)
        # print(self.avg_pool(x).shape)#torch.Size([4, 32, 1, 1, 1])
        y = self.fc(y).view(b, c, 1, 1, 1)
        # print(y.shape,y.expand_as(x).shape)#torch.Size([4, 32, 1, 1, 1]) torch.Size([4, 32, 32, 128, 128])
        return x * y.expand_as(x)


# tensor.expand_as()这个函数就是把一个tensor变成和函数括号内一样形状的tensor，用法与expand（）类似。
# 差别是expand括号里为size，expand_as括号里为其他tensor。
# expand()函数可以将张量广播到新的形状，但是切记以下两点：
# 1. 只能对维度值为1的维度进行扩展，且扩展的Tensor不会分配新的内存，只是原来的基础上创建新的视图并返回；
# 2. 无需扩展的维度请保持维度值不变。


class Encoder(nn.Module):
    """
    A single encoder module consisting of
    单个encoder的组成:
    (1)连续的两个卷积层(例如 BN3d+ReLU+Conv3d)
    参数order可改变encoder的结构
    (2)池化层，可能和普通的(2,2,2)结构不太一样 ，例如，如果体积数据是各向异性的（确保在解码器路径中使用互补scale_factor），然后是DoubleConv模块。

    Args:参数
        in_channels (int): number of input channels
        middle_channels (int): number of middle channels
        out_channels (int): number of output channels
        apply_pooling (bool): if True use pooling
        conv_kernel_size (int or tuple): size of the convolving kernel
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        conv_layer_order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    # encoder_1 = Encoder(in_channels=in_channels, middle_channels=16, out_channels=32, apply_pooling=False, conv_kernel_size=3, pool_kernel_size=2, pool_type='max', conv_layer_order=layer_order, num_groups=8, padding=1, stride=1)

    def __init__(
        self,
        in_channels,  # 输入通道数
        middle_channels,  # 中间层通道数
        # 因为一个encoder有两个卷积层，第一层卷积层out_channle=middle,第二层in_channle=middle
        out_channels,
        apply_pooling=True,  # 是否应用池化
        conv_kernel_size=3,  # 卷积核的size
        pool_kernel_size=2,
        pool_type="max",  # 池化方式 max/avg
        conv_layer_order="gcr",  # 决定encoder的结构 /各个层的顺序
        num_groups=8,  # 几个groups
        padding=1,
        stride=1,
        use_dsc=False,
    ):
        super(Encoder, self).__init__()
        assert pool_type in ["max", "avg"]
        if apply_pooling:
            if pool_type == "max":
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        self.use_dsc = use_dsc
        # conv1
        self.conv1 = SingleConv(
            in_channels,
            middle_channels,
            conv_kernel_size,
            conv_layer_order,
            num_groups,
            padding=padding,
            stride=stride,
        )
        if use_dsc:
            # self.conv2x = DCN_Conv(
            #     in_ch=middle_channels,
            #     out_ch=middle_channels,
            #     kernel_size=conv_kernel_size,
            #     extend_scope=1.0,
            #     morph=0,
            #     if_offset=True,
            # )
            self.conv2x = DCN_Conv(
                in_ch=middle_channels,
                out_ch=middle_channels,
                kernel_size=conv_kernel_size,
                extend_scope=1.0,
                morph=0,
                if_offset=True,
            )
            self.conv2z = DCN_Conv(
                in_ch=middle_channels,
                out_ch=middle_channels,
                kernel_size=conv_kernel_size,
                extend_scope=1.0,
                morph=2,
                if_offset=True,
            )
            # conv3
            self.conv2 = SingleConv(
                middle_channels * 2,
                middle_channels,
                conv_kernel_size,
                conv_layer_order,
                num_groups,
                padding=padding,
                stride=stride,
            )
        else:
            self.dilation_conv = SingleConv(
                middle_channels,
                middle_channels,
                (3, 3, 3),
                conv_layer_order,
                num_groups,
                padding=(4, 4, 4),
                stride=stride,
                dilation=4,
            )
        self.conv3 = SingleConv(
            middle_channels,
            out_channels,
            conv_kernel_size,
            conv_layer_order,
            num_groups,
            padding=padding,
            stride=stride,
        )
        # # 相比前两层padding=(4,4,4)且dilation=4
        # # We adopted dilated convolutions as they enlarged the feature extraction area without increasing the size of the model.
        # self.dilation_conv = SingleConv(middle_channels, middle_channels, (3, 3, 3), conv_layer_order,
        #                                 num_groups, padding=(4, 4, 4), stride=stride, dilation=4)

        self.att = AttModule(channel=middle_channels)  # SEnet
        # 3代表cat了三个特征图

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.conv1(x)
        if self.use_dsc:
            # x_2x_0 = self.conv2x(x)
            x_2x_0 = self.conv2x(x)
            x_2z_0 = self.conv2z(x)

            x_2 = torch.cat([x_2x_0, x_2z_0], dim=1)
            x = x + self.att(self.conv2(x_2))
        else:
            x = x + self.att(self.dilation_conv(x))
        x = self.conv3(x)

        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    由上采样层组成的单模块_for decoder path
    可选ConvTranspose3d/nearest neighbor interpolation，然后是一个基本模块(如双卷积encoder模块)

    (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (e.g. double conv like encoder).
    Args:
        in_channels (int): number of input channels
        middle_channels (int): number of middle channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the conv kernel
        deconv_kernel_size (int or tuple): size of the deconv kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is_deconv (bool):
    """

    # decoder_1 = Decoder(in_channels=256, upsample_out_channels=256, conv_in_channels=384,
    #                     conv_middle_channels=128, out_channels=128, conv_kernel_size=3,
    #                     conv_layer_order=layer_order, num_groups=8, conv_padding=1,
    #                     conv_stride=1, deconv_kernel_size=4, deconv_stride=(2, 2, 2), deconv_padding=1)
    # layer_order='gcr'
    def __init__(
        self,
        in_channels,
        upsample_out_channels,
        conv_in_channels,
        conv_middle_channels,
        out_channels,
        conv_kernel_size=3,
        conv_layer_order="gcr",
        num_groups=8,
        conv_padding=1,
        conv_stride=1,
        deconv_kernel_size=4,
        deconv_stride=(2, 2, 2),
        deconv_padding=1,
        use_dsc=False,
    ):
        super(Decoder, self).__init__()
        # deconv
        self.use_dsc = use_dsc
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            upsample_out_channels,
            kernel_size=deconv_kernel_size,
            stride=deconv_stride,
            padding=deconv_padding,
        )

        # concat joining
        self.joining = partial(self._joining, concat=True)
        # 是 Python 标准库中的一个函数式编程工具，它可以用来“冻结”函数的一部分参数，生成一个新的可调用对象。
        # 这个新的可调用对象可以像原函数一样接收剩余的参数，并将“冻结”部分的参数自动传递给原函数
        # 所以这里self.joining函数只是self._joining的换皮，只是新的函数的concat永久冻结成了True

        self.conv1 = SingleConv(
            conv_in_channels,
            conv_middle_channels,
            conv_kernel_size,
            conv_layer_order,
            num_groups,
            padding=conv_padding,
            stride=conv_stride,
        )
        if self.use_dsc:
            self.conv2x = DCN_Conv(
                in_ch=conv_middle_channels,
                out_ch=conv_middle_channels,
                kernel_size=conv_kernel_size,
                extend_scope=1.0,
                morph=0,
                if_offset=True,
            )
            # self.conv2y = DCN_Conv(
            #     in_ch=conv_middle_channels,
            #     out_ch=conv_middle_channels,
            #     kernel_size=conv_kernel_size,
            #     extend_scope=1.0,
            #     morph=1,
            #     if_offset=True,
            # )
            self.conv2z = DCN_Conv(
                in_ch=conv_middle_channels,
                out_ch=conv_middle_channels,
                kernel_size=conv_kernel_size,
                extend_scope=1.0,
                morph=2,
                if_offset=True,
            )

            self.conv2 = SingleConv(
                2 * conv_middle_channels,
                conv_middle_channels,
                conv_kernel_size,
                conv_layer_order,
                num_groups,
                padding=conv_padding,
                stride=conv_stride,
            )

        else:
            self.dilation_conv = SingleConv(
                conv_middle_channels,
                conv_middle_channels,
                (3, 3, 3),
                conv_layer_order,
                num_groups,
                padding=(4, 4, 4),
                stride=conv_stride,
                dilation=4,
            )
        self.att = AttModule(channel=conv_middle_channels)
        # conv2
        self.conv3 = SingleConv(
            conv_middle_channels,
            out_channels,
            conv_kernel_size,
            conv_layer_order,
            num_groups,
            padding=conv_padding,
            stride=conv_stride,
        )

        #

    def forward(self, encoder_features, x):
        # print(x.shape)#torch.Size([4, 256, 4, 16, 16])#注意3D图像不再是HxW,而是XYZ，所有这里三个参数都变了
        x = self.upsample(x)
        # print(x.size())#torch.Size([4, 256, 8, 32, 32])
        x = self.joining(encoder_features, x)

        x = self.conv1(x)
        # 3dU-Net的concat部分
        if self.use_dsc:
            x_2x_0 = self.conv2x(x)
            # x_2y_0 = self.conv2y(x)
            x_2z_0 = self.conv2z(x)
            x_2 = torch.cat([x_2x_0, x_2z_0], dim=1)
            # x_2 = torch.cat([x_2y_0, x_2z_0], dim=1)
            x = x + self.att(self.conv2(x_2))
        else:
            x = x + self.att(self.dilation_conv(x))
        x = self.conv3(x)

        return x

    # @staticmethod 是一个装饰器，用于将函数转化为静态方法
    # 静态方法在/home/cs22-wangc/now/NaviAirway/aaa_me/a.ipynb中进行了详细的说明
    # 把它看成concat永远为True即可

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            # print(11111)只触发这个
            return torch.cat((encoder_features, x), dim=1)
        else:
            # print(2222)
            return encoder_features + x


class SegAirwayModel(nn.Module):
    def __init__(self, in_channels, out_channels, layer_order="gcr", **kwargs):
        super(SegAirwayModel, self).__init__()

        # Adjusted encoder layers with reduced channels
        encoder_1 = Encoder(
            in_channels=in_channels,
            middle_channels=8,  # Reduced
            out_channels=16,  # Reduced
            apply_pooling=False,
            conv_kernel_size=3,
            pool_kernel_size=2,
            pool_type="max",
            conv_layer_order=layer_order,
            num_groups=4,  # Adjusted for fewer channels
            padding=1,
            stride=1,
            use_dsc=True,
        )

        encoder_2 = Encoder(
            in_channels=16,  # Adjusted
            middle_channels=16,  # Reduced
            out_channels=32,  # Reduced
            apply_pooling=True,
            conv_kernel_size=3,
            pool_kernel_size=2,
            pool_type="max",
            conv_layer_order=layer_order,
            num_groups=4,
            padding=1,
            stride=1,
            use_dsc=False,
        )

        encoder_3 = Encoder(
            in_channels=32,  # Adjusted
            middle_channels=32,  # Reduced
            out_channels=64,  # Reduced
            apply_pooling=True,
            conv_kernel_size=3,
            pool_kernel_size=2,
            pool_type="max",
            conv_layer_order=layer_order,
            num_groups=8,
            padding=1,
            stride=1,
            use_dsc=False,
        )

        encoder_4 = Encoder(
            in_channels=64,  # Adjusted
            middle_channels=64,  # Reduced
            out_channels=128,  # Reduced
            apply_pooling=True,
            conv_kernel_size=3,
            pool_kernel_size=2,
            pool_type="max",
            conv_layer_order=layer_order,
            num_groups=8,
            padding=1,
            stride=1,
            use_dsc=False,
        )

        self.encoders = nn.ModuleList([encoder_1, encoder_2, encoder_3, encoder_4])

        # Adjusted decoder layers with reduced channels
        decoder_1 = Decoder(
            in_channels=128,  # Adjusted
            upsample_out_channels=128,  # Adjusted
            conv_in_channels=192,  # Reduced
            conv_middle_channels=64,  # Reduced
            out_channels=64,  # Reduced
            conv_kernel_size=3,
            conv_layer_order=layer_order,
            num_groups=8,
            conv_padding=1,
            conv_stride=1,
            deconv_kernel_size=4,
            deconv_stride=(2, 2, 2),
            deconv_padding=1,
            use_dsc=False,
        )

        decoder_2 = Decoder(
            in_channels=64,  # Adjusted
            upsample_out_channels=64,  # Adjusted
            conv_in_channels=96,  # Reduced
            conv_middle_channels=32,  # Reduced
            out_channels=32,  # Reduced
            conv_kernel_size=3,
            conv_layer_order=layer_order,
            num_groups=4,
            conv_padding=1,
            conv_stride=1,
            deconv_kernel_size=4,
            deconv_stride=(2, 2, 2),
            deconv_padding=1,
            use_dsc=False,
        )

        decoder_3 = Decoder(
            in_channels=32,  # Adjusted
            upsample_out_channels=32,  # Adjusted
            conv_in_channels=48,  # Reduced
            conv_middle_channels=16,  # Reduced
            out_channels=16,  # Reduced
            conv_kernel_size=3,
            conv_layer_order=layer_order,
            num_groups=4,
            conv_padding=1,
            conv_stride=1,
            deconv_kernel_size=4,
            deconv_stride=(2, 2, 2),
            deconv_padding=1,
            use_dsc=False,
        )

        self.decoders = nn.ModuleList([decoder_1, decoder_2, decoder_3])

        # Final 1x1 convolution to match the number of output channels
        self.final_conv = nn.Conv3d(
            in_channels=16,  # Adjusted
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        # print("intial shape", x.shape)
        i = 0
        for encoder in self.encoders:
            x = encoder(x)
            # print(f"the {i} encoder shape is {x.shape}")
            i += 1
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        j = 0
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

            # print(f"the {j} decoder shape is {x.shape}")
            j += 1
        x = self.final_conv(x)
        # print("final conv ", x.shape)
        x = self.final_activation(x)
        # print("**************************************************")
        return x

    def model_info(self):
        message = "本模型为model_arch_e0_yz.py,只在encoder0上加dsc模块,通道数为model_arch的一半,同时在xz轴进行dsc"
        flag = "dsc"
        return message, flag
