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
[fPath,fName,fExt] = fileparts(file_name);
assert(numel(logs) > 0, ['No log found. Checking for all PIDS: ' log2str(allPids)]);
offset = numel([fName fExt])+1;
val = -Inf;
idx = 0;
% Find newest entry
for i = 1:numel(logs)
    logs(i).name(offset)
    nVal = str2double(logs(i).name([offset:offset+7, offset+9:offset+14]));
    if nVal > val
        val = nVal;
        idx = i;
    end
end
copyfile([fPath '/' logs(idx).name],file_name);
delete([fPath '/' logs(idx).name]);

end

function str = log2str(val)
assert(islogical(val),'Input value must be logical');
if val
    str = 'Yes';
else
    str = 'No';
end

end