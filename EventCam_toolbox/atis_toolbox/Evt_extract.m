function BE=Evt_extract(AE,type)

%Type event extraction from a ATIS structure file. Mainly used for
%separating DVS from GL signal.
%streams in AE structure:
%AE.x       uint16    column
%AE.y       uint16    row
%AE.t      double    time in s
%AE.source  uint16    polarity
%AE.type    uint16    tD if 1, GL if 0

L=find(AE.type==type);
BE.x=AE.x(L);
BE.y=AE.y(L);
BE.t=AE.t(L);
BE.source=AE.source(L);

    