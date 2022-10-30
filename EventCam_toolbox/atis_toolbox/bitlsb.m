function Y=bitlsb(X);

%/***********************************************************************
% *     PROJECT: Silicon Retina Leitprojekt
% *   COPYRIGHT: ARC Seibersdorf Research GmbH, Business Unit ITN
% *    $RCSfile: bitlsb.m,v $
% *   $Revision: 1.1 $
% *      AUTHOR: M.Litzenberger
% *  SUBSTITUTE: 
% *    LANGUAGE: Matlab
% *
% * DESCRIPTION: Pos = bitlsb(number)
%                Returns the position of the least significant bit (LSB) in an INT-number.
%                Count starts at 0!
% *
% *    PARAMETER
% *          IN: number .. binary data
% *         OUT: Pos    .. position of the LSB
% *
% * $Log: bitlsb.m,v $
% * Revision 1.1  2005/12/21 11:58:42  litzenbergerm
% * Migration from VSS to CVS. Use VSS to retrive all earlier versions of this file.
% *
% 
% 3     16.02.05 15:47 Litzenbergerm
% Added standard header
% *
% ************************************************************************/

XX=X; Y=0;
while ~bitand(1,XX);
    XX=bitshift(XX,-1);
    Y=Y+1;
end
return
