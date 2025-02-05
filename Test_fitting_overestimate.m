originalFolder = pwd;

targetFolder = './spherical_data_fitting';
addpath('./Mh');

manoptFile = './manopt';
cd(manoptFile);
run('importmanopt');


cd('..');
cd(targetFolder);
experiment_overestimate(0.1);


cd(originalFolder);