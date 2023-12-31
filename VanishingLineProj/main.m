% Vanishing Points detection algorithm.
% IPOL SUBMISSION "Vanishing Point Detection in Urban Scenes Using Point Alignments"
% 
% Version 0.6, July 2015
%
% This version includes the optional use the algorithm by Figueiredo and 
% Jain, Unsupervised learning of finite mixture models, to quickly obtain 
% cluster candidates.
%
% Copyright (c) 2013-2015 Jose Lezama <jlezama@gmail.com>
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as
% published by the Free Software Foundation, either version 3 of the
% License, or (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU Affero General Public License for more details.
% 
% You should have received a copy of the GNU Affero General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.

clear
close all

img_in =  '..\pics\azrielli.jpg'; % input image

% Extract file name and extension
[~, name, ext] = fileparts(img_in);

% Create folder path
folder_path = fullfile('..\', 'input_files', name);

% Check if the folder exists, and create it if not
if ~exist(folder_path, 'dir')
    mkdir(folder_path);
end
folder_out = folder_path; % output folder

manhattan = 1;
acceleration = 0;

focal_ratio = 1.08;

params.PRINT = 1;
params.PLOT = 1;

horizon = detect_vps(img_in, folder_out, manhattan, acceleration, focal_ratio, params);
