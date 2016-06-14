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
      if nargin == 2
          layersIp = {};
          layersConv = {};
      end

      % Generate mapping lists. All layers with matching names and number of parameters
      % are added to the current mapping
      for name = netIp.layer_names'
          layer = netIp.layers(char(name));
          if numel(layer.params) > 0 && any(strcmp(netConv.layer_names,char(name)))
              layer2 = netConv.layers(char(name));
              for i = 1:numel(layer.params)
                  if prod(layer.params(i).shape) ==...
                          prod(layer2.params(i).shape)
                      layersIp = [layersIp,name];
                      layersConv = [layersConv,name];
                  end
              end
          end
      end
      assert(numel(layersIp) == numel(layersConv),'Number of original and new layers should be equal');
      for i = 1:numel(layersIp)
        for j = 1:numel(netConv.layers(layersConv{i}).params)
          data = netIp.params(layersIp{i},j).get_data();
          if numel(netConv.params(layersConv{i},j).shape) > 1 
            data = reshape(data,netConv.params(layersConv{i},j).shape);
          end
          netConv.params(layersConv{i},j).set_data(data);
        end
      end
    end
  end
end
