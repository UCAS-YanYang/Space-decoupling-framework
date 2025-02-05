originalFolder = pwd;

targetFolder = './spherical_data_fitting';
addpath('./Mh');

manoptFile = './manopt';
cd(manoptFile);
run('importmanopt');


cd('..');
cd(targetFolder);
experiment_exact(0.01);

cd(originalFolder);