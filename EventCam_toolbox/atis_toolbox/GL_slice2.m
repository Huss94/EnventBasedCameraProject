function [It,Time,evt]=GL_slice2(evt,is,ie,timeslot,mode)

%Function computing graylevel from the ATIS events and build frames at
%1/timeslot fps.
%Reminder: Streams in AE structure:
%evt.x       uint16    column
%evt.y       uint16    row
%evt.t      double    time in s
%evt.gl      double    time in s
%
%Example of use:
%
%[It,T]=GL_slice(ev_struct,start_idx,end_idx,timeslot,mode); 
%
%[It,T]=GL_slice(ev_struct,1,length(AE.t),1e-3,0) 
%creates frames each 1ms from the entire AE stuct.
%
%Inputs
%AE_struct: atis data
%start_idx: index of event in AE to start the extraction
%end_start: last event
%timeslot : duration for frame integration
%mode     : mode to select for frame generation (mode =0) or fake dvs
%stream (mode =1).
%
%Outputs
%It : A 3D array storing graylevel with the third axis as the time. To be
%used only for short time with mode=1.
%Time : timestamp of each frame.
%evt : fake dvs events. Dates of each successful gl integration.

%==========================================================================
%   Codes updated the 14th march 2012
%   Author: s. ieng
%   Add comment here if updating code.
%   New update: 5th april 2012 - add the gl field
%==========================================================================



%stream length
sz=ie-is;

%size of the sensor
%mx1=max(AE.x(:)); my1=max(AE.y(:));
mx=304; my=240;
%mix1=min(AE.x(:)); miy1=min(AE.y(:));
mix=1; miy=1;

%Buffers initialization
idbuffer=zeros(my,mx);
intensitybuffer=zeros(my,mx);

%cells' state array. A cell is the set of values defined in (x,y) at t
State=zeros(my,mx);                 %state matrix of the pixels: first level is the state
State(:,:,1)=-1;                    %i.e. 0 if an integration starts and ~= otherwise.

%global output frame
I=zeros(my,mx); 

%time slice centered on [l,c] of the AE: do not generate for too long seq.
It=[]; evt=[]; Time=[];
fmax=2000; %max frame number

if mode == 0
    It=zeros(my,mx,fmax); 
    Time=zeros(fmax,1);
else
    %field initialization.
    evt.x=AE.x;
    evt.y=AE.y;
    evt.t=AE.t;
    evt.gl=AE.t;
end

%counters
k=1; j=1;

%init time:
t0=AE.t(is);



%loop over the entire stream
for i=is:ie
    
    %check if it is a GL event
    if AE.type(i)==0
        
        %check if the event starts the measurement
        if AE.source(i)==0                      
            State(AE.y(i),AE.x(i))=0;
            idbuffer(AE.y(i),AE.x(i))=i;
            
        else                                     %means AE.source=1 i.e. end of the measurement
            if State(AE.y(i),AE.x(i))==0         %check if the measurement succeeds
                
                
                %get the index of the measurement starting time.
                id=idbuffer(AE.y(i),AE.x(i));       
                
                %buffer handling
                intensitybuffer(AE.y(i),AE.x(i))=(AE.t(i)-AE.t(id)); %compute the graylevel (inverse).
                
                %resetting to non measurement state
                State(AE.y(i),AE.x(i))=-1;
                
                I(AE.y(i),AE.x(i))=1/intensitybuffer(AE.y(i),AE.x(i));
                
                if mode==1
                    %It(k,AE.x(i))=I(AE.y(i),AE.x(i));
                %else
                    %struct of graylevel
                    evt.x(k)=AE.x(i);
                    evt.y(k)=AE.y(i);
                    evt.t(k)=AE.t(id);
                    evt.gl(k)=I(AE.y(i),AE.x(i));
                end
                
                k=k+1;
            end
            
        end
    end
    
    if mode == 0
    %frames generation for each timeslot
        if (AE.t(i)-t0)>timeslot

            Time(j)=AE.t(i);
            It(:,:,j)=I; 
            j=j+1
            t0=AE.t(i);
        end
    end
    

end
%It(k:end,:)=[];

%arrays reshape.
if mode == 0
    It(:,:,j:end)=[];
else
    %resize and reshape structure
    evt.x(k:end)=[]; 
    evt.y(k:end)=[];
    evt.t(k:end)=[];
    evt.gl(k:end)=[];
    
    %sort in time ascending order
    [~,lst]=sort(evt.t);
    evt.x=evt.x(lst);
    evt.y=evt.y(lst);
    evt.t=evt.t(lst);
    evt.gl=evt.gl(lst);
end


                
                