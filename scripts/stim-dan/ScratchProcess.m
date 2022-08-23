addpath(genpath('C:\TDT\TDTMatlabSDK')); % Add TDT files to path
addpath(genpath('C:\Dropbox\Projects\BMI-FES\Code')); 

path = {'C:\Dropbox\Projects\BMI-FES\Chronic Wired\D1\D1-220603-142830', ... % spinal caud + rostral
    'C:\Dropbox\Projects\BMI-FES\Chronic Wired\D1\D1-220603-143459', ... % 
    'C:\Dropbox\Projects\BMI-FES\Chronic Wired\D1\D1-220603-143634', ... % spinal + GA
    'C:\Dropbox\Projects\BMI-FES\Chronic Wired\D1\D1-220603-144203', ... % rostral + TA
    'C:\Dropbox\Projects\BMI-FES\Chronic Wired\D1\D1-220603-144751', ... % L4 + rostral alternating (left first then right)
    'C:\Dropbox\Projects\BMI-FES\Chronic Wired\D1\D1-220603-145511'}; % spinal alone

% tdt struct
spinalcaud = TDTbin2mat(path{3});
% build table of all epocs
spinalcaudTable = buildEpocTable(spinalcaud);


channel_map = {' ',' ',' ',' ','RostralL'};


num_cam = 2;

%% load time vector for 42830
p = 'C:\Dropbox\Projects\BMI-FES\Chronic Wired\D1\D1-220603-142830\live_videos\TIMESTAMPS_cam2_D1-220603-142830_200f-11e100g.csv';
cam2_kin_time = readtable(p,'TextType','char');

% Use the import data function...read table sucks
cam2_kin_time = TIMESTAMPScam2D1220603142830200f11e100g;
cam2_kin_time_norm = cam2_kin_time - cam2_kin_time(1)
%% load kinematics

spinalcaudkin_files = dir('C:\Dropbox\Projects\BMI-FES\Chronic Wired\D1\combined_vids\*144203*csv');
kin_variables = ["pelvistop_x" "pelvistop_y" "hip_x" "hip_y" "pelvisbot_x" "pelvisbot_y" ...
    "knee_x" "knee_y" "ankle_x" "ankle_y" "MTP_x" "MTP_y" "toe_x" "toe_y"];
vid_path = 'C:\Dropbox\Projects\BMI-FES\Chronic Wired\D1\combined_vids\';

for cam = 1:num_cam
rawcsvtable = readtable([vid_path spinalcaudkin_files(cam).name],'TextType','string');
rawcsvarray = cell2mat(cellfun(@str2num,table2array(rawcsvtable(3:end,[2 3 5 6 8 9 11 12 14 15 17 18 20 21])),'UniformOutput',0));

kinematics{cam} = array2table(rawcsvarray,'VariableNames',kin_variables);
end

%% Calculate angles

% knee, ankle(center), MTP. Order matters.this is for left ankle
ankle_angle = jointAngle([kinematics{2}.knee_x kinematics{2}.knee_y], ...
    [kinematics{2}.ankle_x kinematics{2}.ankle_y], ...
    [kinematics{2}.MTP_x kinematics{2}.MTP_y]);
limb_angle = jointAngle([kinematics{2}.pelvistop_x kinematics{2}.pelvistop_y], ...
    [kinematics{2}.hip_x kinematics{2}.hip_y], ...
    [kinematics{2}.ankle_x kinematics{2}.ankle_y]);


%% Plot the whole file

kin_fs = 200;
kin_ts = 1/200;

% 8 second filler at the start
kin_start_filler = zeros(round(spinalcaud.epocs.P1SC.onset/kin_ts),1);

kin_full = [kin_start_filler; limb_angle'];

% kinematics

kin_time = linspace(0,kin_ts*length(kin_full),length(kin_full));
%spinalcaud.epocs.P1SC.onset:kin_ts:kin_ts*height(kinematics{1}); % This time vector starts when the pulse is sent to cameras (ie 8 sec)



figure; 
% plot the kinematics
plot(kin_time,kin_full)
hold on;

% plot stim onsets
for s = 1:height(spinalcaudTable)
    xline(spinalcaudTable.onset(s),'r')
    
    
end


%% plot individual trials in the table
trial = 1;
time_poststim = 1; % number of seconds AFTER stim you want to plot
time_prestim = 1; % number of seconds before stim you want to plot

% stimulation compliance plotting parameters
stim_Fs = spinalcaud.streams.Bnk1.fs;
stim_Ts = 1/spinalcaud.streams.Bnk1.fs;
stim_time = 0:stim_Ts:length(spinalcaudTable.compliance{trial})*stim_Ts - stim_Ts;

% kinematics parameters
[m,kin_curr_trial_index]=min(abs(kin_time - spinalcaudTable.onset(trial))); % Find where in kinematics sample this trial is

% Make kinematics time vector
kin_currtrial_time = spinalcaudTable.onset(trial) - time_prestim:kin_ts:time_poststim+spinalcaudTable.onset(trial);

figure;
subplot(5,1,5); % stimulation
hold on; title('Stimulation')
plot(stim_time,spinalcaudTable.compliance{trial}(1,:));
a = gca;
a.XLim = [0-time_prestim time_poststim];

subplot(5,1,2); % toe y
hold on; title('Toe Y')
plot(kin_currtrial_time,kinematics{cam}.toe_y(kin_curr_trial_index:kin_curr_trial_index + length(kin_currtrial_time)-1)); hold on;
a= gca;
a.XLim = [spinalcaudTable.onset(trial) time_poststim+spinalcaudTable.onset(trial)];
a.YLim = [0 300];
subplot(5,1,3); % MTP
hold on; title('MTP y')
plot(kin_currtrial_time,kinematics{cam}.MTP_y(kin_curr_trial_index:kin_curr_trial_index + length(kin_currtrial_time)-1));
a= gca;
a.XLim = [spinalcaudTable.onset(trial) time_poststim+spinalcaudTable.onset(trial)];
a.YLim = [0 300];
subplot(5,1,4); % Knee
hold on; title('Knee y')
plot(kin_currtrial_time,kinematics{cam}.knee_y(kin_curr_trial_index:kin_curr_trial_index + length(kin_currtrial_time)-1));
a= gca;
a.XLim = [spinalcaudTable.onset(trial) time_poststim+spinalcaudTable.onset(trial)];
a.YLim = [0 300];
subplot(5,1,1); % Hip
hold on; title('hip y')
plot(kin_currtrial_time,kinematics{cam}.hip_y(kin_curr_trial_index:kin_curr_trial_index + length(kin_currtrial_time)-1));
a= gca;
a.XLim = [spinalcaudTable.onset(trial) time_poststim+spinalcaudTable.onset(trial)];
a.YLim = [0 300];
% movie


