
function [serie]=vizu_activity_ROI(percent_display,I,J,type_event,Tps)
% --------------------------------------------------------------------
%
% vizu_activity: 
%
%   function to vizualize (quickly) the content of an event 
%   sequencis and select a subsequencis
%
% vizu_activity(I,J,type_events,Tps)
% 
%    IN: all these input arguments are optional (see below)
%
%       (I,J) = coordinates of the events (I,J = 1xN vectors, N=number of events)
%       type of events = type of the events (1xN vector) % not really use !
%               'dvs' -> polarities
%               'atis' -> (pseudo) gray level
%       this is not really used for now so commenting the whole thing would
%       not be a problem.
%       Tps = times of the events (1xN vector)
%
%
% % Use: 
% %   * Normal use is to call the function with no inputs and select the DVS
% %   / ATIS file, the function will automatically do the rest.
% %   * the normal structure consists of a .x vector, .y vector, .ts vector and .p vector.
% %   If your structure is different, please input the vectors manually.
% %   * a splitted figure is shown (Up: a "synchrony image" of the events,
% %   down: a measure of the activity = 1/(diff(Tps)))
% %   * move twice in the activity subplot to select a time window
% %   (beginning and THEN ending, else it will fail). It will prompt a new
% %   window and 
% %   
% Update:
%   12 juin 2013: in order to work with bigger "image", an automatic
%                 adaptation of the number of events shown...
%   
% 
% comments/todos/bugs: 
%
% - the input arguments are optional: if they are empty, you can select 
%   an "aer file" to read...
% - a test and repair function (search_and_cut_error.m) is proposed
%   to select a correct subsequencis if the timing data are not strictly
%   increasing - todo: write a better explanation
% - the "synchrony image" is computed using a Quiroya-like synchrony measure
% - the polarities/gray levels of the events are not vizualized (in this version)
% - the duration of the short period is automatically computed (the
%   sequencis is splitted in 1000 bins) - todo: add it as an input argument 
% - todo: to write better and clearer help + title, label,... of the figure
%   Please accept my apologies for this preliminary version!
% - be careful: this version closes all the matlab figure.
%
% depending: search_and_cut_error.m, mise_en_memoire.m ,
% load_atis_data_2_le_retour.m
%            + tobi's matlab library (aer_toolbox)
%
% Version 1.0 (18/04/2020) 
% Written by Xavier Clady, xavier[dot]clady[at]upmc[dot]fr
%
% Modified By Adam Heriban, heriban[at]crans[dot]org
% --------------------------------------------------------------------
%


% init some variables
numero=[];
debut=1;
testOK=0;
x_souris=0.5;
y_souris=0.5;
x_coord=0;
y_coord=0;
SERIE.x = 0;
SERIE.y = 0;
SERIE.ts = 0;
SERIE.p = 0;

% test of the input arguments
if (nargin<1),
    percent_display=70;
end
if (nargin<=1)
    
    % Select and Read the event sequencis
    disp('-> Read new sequencis: ');
    
    if nargin<=1,
    [filename,filepath,filterindex]=uigetfile('*','Select recorded retina data file');
    
    if filename==0 
        return; 
    end
    end
    path1 = pwd;
    f=fopen([filepath,filename],'r');
    line=native2unicode(fgets(f));
    if line(1) == '#'  % DVS file header
        answer=menu('Your file is recognized as an DVS file. Is it correct ?','yes','no');
        if answer==2,
			disp('Sorry, this file format is not compatible with this code');
			quit CANCEL;
		end
        [I,J,type_event,Tps,sequencis.filename]=mise_en_memoire(filename,filepath); % this is a program that loads dvs files

    elseif line(1) == '%' % ATIS file header
		answer=menu('Your file is recognized as an ATIS file. Is it correct ?','yes','no');
		if answer==2,
			disp('Sorry, this file format is not compatible with this code');
			quit CANCEL;
		end

        serie = load_atis_data_2_le_retour(filename,filepath); % new verrsiion of load_atis_data
        I = serie.x;
        J = serie.y;
        Tps = serie.ts;
        type_event = serie.p;

    elseif line(1:6) == 'MATLAB' % already processed matlab file
		answer=menu('Your file is recognized as an already processed matlab file. We assume that vectors are named x, y, p, ts. If not, rename them or change the program variables. Is it the case ?','yes','no');
		if answer==2,
		disp('Sorry, this file format is not compatible with this code');
		quit CANCEL;
		end

        str = input('What is the name of the structure ? (empty means vectors are already here) :','s'); % because this is a matlab file, we do not know the name of everything
        cd(filepath);
        S = load(filename);
        found = 0;
        tries = 0;
        % This huge loop is used to find the data inside the matlab file
        % (at least the structure)
        % We assume that the vectors are named x, y, ts, and p. If not,
        % change their name
        if strcmp(str, '');
                serie = S;
        else
            while found == 0
                if isfield(S,str)
                    serie = S.(str);
                    if isfield(serie,'x')
                        found =1;
                        disp('Ok, we found the sequence');
                    elseif tries == 5  % after five iterations we will consider that the file is useless
                        disp('This is the 5th iteration. sorry but the program will now crash (probably). Please load the vectors manually');
                        found = 1;
                    else
                        tries = tries + 1;
                        disp('It seems this is not a sequence. If the sequence is further inside the file, please enter the next step. Else, stop the program');
                        str = input('What is the name of the structure ? :','s');
                    end
                else
                    disp('Wrong name, try again');
                    str = input('What is the name of the structure ? :','s');

                end
            end
        end
        I = serie.x;
        J = serie.y;
        Tps = serie.ts;
        type_event = serie.p;

    else 
        disp('unknown file type, please convert it manually to vectors');

    end
    cd(path1);
    
    % in case of temporal error in data (often in the
    % beginning of the sequence)
    disp('-> Test the sequencis (and repair it if necessary): ')
    [debut,fin]=search_and_cut_error(Tps);
    if (debut==1 && fin==length(fin))  
        disp('no problem detected')
    else
        disp('Warning: the sequencis has been cuted (some timing problem)')
        I=I(debut:fin);
        J=J(debut:fin);
        Tps=Tps(debut:fin);
        type_event=type_event(debut:fin);
        Tps=double(Tps); % ne sert pas ??? grand chose !
    end
    nb_rows=max(I);         % we assume there that all events are between 0 and a maximum
    nb_columns=max(J);

elseif nargin == 5      % right number of inputs
    nb_rows=max(I);         % we assume there that all events are between 0 and a maximum
    nb_columns=max(J);    
else
    error('Bad use of the function -> help vizu_activity') % to many inputs
    help vizu_activity_ROI
end

% preprocessing
disp('-> Process the data (wait please) : ')
serie.x = I;
serie.y = J;
serie.ts = Tps;
serie.p = type_event;
% just to quickly make sure there is nothing with a 0 coordinate (crashes
% the whole thing)
Z1 = (I == 0);
Z2 = (J == 0);

I = I + Z1;
J = J + Z2;


positions=sub2ind([nb_rows nb_columns],I,J); 

nb_events_shown=ceil((nb_rows*nb_columns)/percent_display);

subseq_border_t=[];
subseq_border_space = [];
subseq_border_space2 = [];


selection=1:ceil(length(Tps)/nb_events_shown):length(Tps);
Tps_short=Tps(selection);
activity_inv=conv(diff(Tps_short),ones(2,1),'same');
SIZE = size(I);
if SIZE(1,1) == 1
    
    activity=1./activity_inv;
    activity=[0 activity/max(activity)];
else
    activity=1./activity_inv;
    activity=[0;activity/max(activity)];    
end    


% displaying
figlaser=figure('name','Quick Vizualization and Activity',...
    'numbertitle','off',...
    'tag','interface',...
    'WindowButtonMotionFcn',@motion,...
    'WindowButtonDownFcn',@down,...
    'WindowButtonUpFcn',@up);hold off;
subplot('Position',[0.05 0.2 0.9 0.75]),  
colormap('gray');
    %axis('image');
subplot('Position',[0.05 0.05 0.9 0.1]),
    plot(selection,activity,'r');
    axis([0  length(Tps) 0 1])

%axislaser=gca;
% wait the mouse click (the user selection OR out)
while(testOK==0)
        pause(0.0001);
end

close all;


function motion (obj,event)
        % Recuperation de la position de la souris et affichage
        % Get the position of the mouse and plot the corresponding synchrony image
        
        figure(figlaser);
        hold on;
        axislaser=gca;
        P=get(axislaser,'CurrentPoint');
        x_souris=P(1,1);
        y_souris=P(1,2);

        numero=ceil(x_souris);
        x_coord = ceil(x_souris * nb_columns / length(Tps));
        y_coord = ceil((9 - y_souris) * nb_rows / 7.6);

        img=zeros(nb_rows,nb_columns);
        pos_inf=max(ceil(x_souris-nb_events_shown),1);
        pos_inf=min(pos_inf,length(Tps));
        
        pos_sup=min(ceil(x_souris)+nb_events_shown,length(Tps));
        pos_sup=max(pos_sup,1);
       
        img(positions(pos_inf:pos_sup))=type_event(pos_inf:pos_sup);
        % display
        subplot('Position',[0.05 0.2 0.9 0.75]),  
            imagesc(img);
            %axis('image')
        subplot('Position',[0.05 0.05 0.9 0.1]),
end

% Callback WindowButtonDownFcn : Appui sur le bouton de la souris
    function down(obj,event)
        etat_souris=1;
        testOK=1;
    end

% Callback WindowButtonUpFcn : Relachement du bouton de la souris
    function up(obj,event)
        etat_souris=0; 
    end

end