# hair-seg
头发分割
输出是512x512单通道的0到1的软分割值， 输入是人脸检测框长宽取大作为base，向左向右分别扩展1.5个base， 向上扩展一个base，向下扩展2.2个base，前提是不超过图片边界，然后resize到512x512，接下来就是之前同样的transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
