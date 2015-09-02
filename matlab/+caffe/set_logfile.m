function set_logfile(file_name)
% set_logfile(file_name)
%   set Caffe's log file

CHECK(ischar(file_name), ...
  'file_name must be a string');

caffe_('set_logfile', file_name);

end