classdef surgery
  % a class for simple network surgery operations
  
  methods (Static)
    function netConv = ip2conv(netIp,netConv,layersIp,layersConv)
      % IP2FCONV Perform surgery to convert a network with
      % InnerProduct layers to Convolution layers using the original
      % weights as kernel weights.
      % The number of parameters for the IP and Conv layers must be
      % equal.
      % Enumerate the names of the IP layers and Conv layers in
      % inputs
      assert(numel(layersIp) == numel(layersConv),'Number of original and new layers should be equal');
      for i = 1:numel(layersIp)
        netConv.params(layersConv{i},1).set_data(reshape(...
          netIp.params(layersIp{i},1).get_data(),...
          netConv.params(layersConv{i},1).shape));

        netConv.params(layersConv{i},2).set_data(...
          netIp.params(layersIp{i},2).get_data());
      end
    end
  end
end