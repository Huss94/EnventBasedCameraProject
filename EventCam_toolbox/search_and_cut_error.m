function [debut,fin]=search_and_cut_error(Tps)
% 
% search_and_cut: 
%
%   search an error in the event timing
%   if error(s) is (are) detected, this function select the 
%   longest subsequencis and send its beginning (=debut) and ending (=fin) 
%   indices
% 
% comment:
%
%   sometimes, the event time is not strictly increasing 
%   (often at the beginning of the sequencis)
% 
% todo : 
%
%   to write a clearer help (please accept my apologies for this one)
% 
% Version 0.9 (03/20/2012) 
% Written by Xavier Clady, xavier[dot]clady[at]upmc[dot]fr
% 

Tps=double(Tps);
diff_tps=diff(Tps);
[position]=find(diff_tps<0);

dimension=size(position);
if dimension(2)<dimension(1),
    position=position';
end

if isempty(position),
    debut=1;fin=length(Tps);
    return;
else
    position=[1,position,length(Tps)];
    [duree]=diff(position);
    [duree_max,indice_max]=max(duree);
    debut=position(indice_max)+1;
    fin=position(indice_max+1);
end


