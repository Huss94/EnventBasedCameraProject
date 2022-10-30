
% lecture d'un fichier ATIS ('..._td.dat')
filename='carre_td.dat';
filepath='./';
serie = load_atis_data_2_le_retour(filename,filepath);
% serie.ts : tableau des temps (en micros)
% serie.x : coord en x
% serie.y : coord en y
% serie.p : polarit? (-1:OFF, 1:ON)

figure,plot(serie.ts*10^-6);xlabel('numero event');ylabel('temps des events (s)')
% 
% % lecture d'un fichier DVS ('....aedat')
% filename='car1.aedat';
% filepath='./';
% serie = load_atis_data_2_le_retour(filename,filepath);
% % serie.ts : tableau des temps (en micros)
% % serie.x : coord en x
% % serie.y : coord en y
% % serie.p : polarit? (-1:OFF, 1:ON)
% 
% figure,plot(serie.ts*10^-6);xlabel('numero event');ylabel('temps des events (s)')