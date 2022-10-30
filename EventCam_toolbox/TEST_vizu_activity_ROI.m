
%% to test vizu_activity_ROI.m
% % Use: 
% %   * Normal use is to call the function with no inputs and select the DVS
% %   / ATIS file, the function will automatically do the rest.
% %   * the normal structure consists of a .x vector, .y vector, .ts vector and .p vector.
% %   If your structure is different, please input the vectors manually.
% %   * a splitted figure is shown (Up: a "synchrony image" of the events,
% %   down: a measure of the activity = 1/(diff(Tps)))
%%    * Click on the window to quit...

serie=vizu_activity_ROI();
%% Select the file "walking_people.aedat"

% serie.ts : tableau des temps (en micros)
% serie.x : coord en x
% serie.y : coord en y
% serie.p : polarit? (-1:OFF, 1:ON)
save('s.mat','-struct', 'serie')
% save("y.mat", serie.x)
% 
% save("p.mat", serie.x)

% figure,plot(serie.ts*10^-6);xlabel('numero event');ylabel('temps des events (s)')




