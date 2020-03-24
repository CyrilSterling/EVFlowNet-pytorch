import torch
from torch import nn

class build_resnet_block(nn.Module):
    """
    a resnet block which includes two general_conv2d
    """
    def __init__(self, channels, layers=2, do_batch_norm=False):
        super(build_resnet_block,self).__init__()
        self._channels = channels
        self._layers = layers

        self.res_block = nn.Sequential(*[general_conv2d(in_channels=self._channels,
                                             out_channels=self._channels,
                                             strides=1,
                                             do_batch_norm=do_batch_norm) for i in range(self._layers)])

    def forward(self,input_res):
        inputs = input_res.clone()
        input_res = self.res_block(input_res)
        return input_res + inputs

class upsample_conv2d_and_predict_flow(nn.Module):
    """
    an upsample convolution layer which includes a nearest interpolate and a general_conv2d
    """
    def __init__(self, in_channels, out_channels, ksize=3, do_batch_norm=False):
        super(upsample_conv2d_and_predict_flow, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._ksize = ksize
        self._do_batch_norm = do_batch_norm

        self.general_conv2d = general_conv2d(in_channels=self._in_channels,
                                             out_channels=self._out_channels,
                                             ksize=self._ksize,
                                             strides=1,
                                             do_batch_norm=self._do_batch_norm,
                                             padding=0)
        
        self.pad = nn.ReflectionPad2d(padding=(int((self._ksize-1)/2), int((self._ksize-1)/2),
                                        int((self._ksize-1)/2), int((self._ksize-1)/2)))#对称padding

        self.predict_flow = general_conv2d(in_channels=self._out_channels,
                                           out_channels=2,
                                           ksize=1,
                                           strides=1,
                                           padding=0,
                                           activation='tanh')

    def forward(self, conv):
        shape = conv.shape
        conv = nn.functional.interpolate(conv,size=[shape[2]*2,shape[3]*2],mode='nearest')#最近邻插值上采样
        conv = self.pad(conv)
        conv = self.general_conv2d(conv)

        flow = self.predict_flow(conv) * 256.
        
        return torch.cat([conv,flow.clone()], dim=1), flow

def general_conv2d(in_channels,out_channels, ksize=3, strides=2, padding=1, do_batch_norm=False, activation='relu'):
    """
    a general convolution layer which includes a conv2d, a relu and a batch_normalize
    """
    if activation == 'relu':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.ReLU(inplace=True)
            )
    elif activation == 'tanh':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.Tanh(),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.Tanh()
            )
    return conv2d

if __name__ == "__main__":
    a = upsample_conv2d_and_predict_flow(1,4)
    b = build_resnet_block(2)
    c = torch.Tensor([[1,2,3,4,5],[4,5,6,2,3],[7,8,9,4,7],[2,3,5,4,6],[4,6,7,4,5]]).reshape(1,1,5,5)
    _, out = a(c)
    out = b(out)
    print(out.shape)
    print(a)
    print(b)
