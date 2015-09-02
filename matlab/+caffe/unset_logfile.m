function unset_logfile( file_name, allPids )
% unset_logfile(file_name)
%   Suspend logging to logfile
if nargin < 2
    allPids = false;
end

% Pass suspension of logging to file on to caffe
caffe_('set_logfile','/tmp/caffe.default');

% Retrieve newest logfile with defined file_name* and copy to file_name
% (removes the datetime and process id from the file name)
if allPids
    logs = dir([file_name '*']);
else
    PID = feature('getpid');
    logs = dir(sprintf('%s*%i',file_name,PID));
end
offset = numel(file_name)+1;
val = -Inf;
idx = 0;
% Find newest entry
for i = 1:numel(logs)
    nVal = str2double(logs(i).name([offset:offset+7, offset+9:offset+14]));
    if nVal > val
        val = nVal;
        idx = i;
    end
end
copyfile(logs(idx).name,file_name);
delete(logs(idx).name);

end