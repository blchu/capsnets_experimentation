class CapsuleLayer(nn.Module):
	def __init__(self,in_capsules,in_channels,out_capsules,unit_size,use_routing,num_iters,use_cuda,filter_size=9,stride=2):
		super(CapsuleLayer, self).__init__()
		self.in_capsules = in_capsules
		self.in_channels = in_channels
		self.out_capsules = out_capsules
		self.unit_size = unit_size
		self.use_routing = use_routing
		self.num_iters = num_iters
		self.use_cuda = use_cuda
		self.filter_size = filter_size
		self.stride = stride

		if self.use_routing:
			#do routing
		else:
			self.conv_capsules = nn.ModuleList([nn.Conv2d(self.in_channels,self.unit_size,self.filter_size,stride) for 
				unit in range(self.out_capsules)])

	def forward(self,x):
		#x=[batch_size,in_channels,width,height]
		capsules = [conv(x) for _,conv in enumerate(self.conv_capsules)]

		capsules = torch.stack(capsules, dim = 1)

		capsules = capsules.view(x.size(0),self.out_capsules,-1)

		return self.squash(capsules,dim=2)

	def squash(vec,dim=2):
		norm = torch.norm(vec,dim,keepdim=True)
		return vec*norm/(1+norm**2)

