clc
clear
close all
%% initial �H��random �y��-------2
coor = 100 ;
points_range = [150, 750; -300, 300; 0, 0];

px = random('unif',points_range(1, 1), sum(points_range(1, :))/2 ,coor/2,1);           %z4��X�b�ܤƶq
py = random('unif',points_range(2, 1), points_range(2, 2),coor/2,1);        %z5��Y�b�ܤƶq
pz = random('unif',points_range(3, 1), points_range(3, 2),coor/2,1); %z6�NZ�b�T�w

px = [px; random('unif',sum(points_range(1, :))/2 + 1, points_range(1, 2),coor/2,1)];           %z4��X�b�ܤƶq
py = [py; random('unif',points_range(2, 1), points_range(2, 2),coor/2,1)];        %z5��Y�b�ܤƶq
pz = [pz; random('unif',points_range(3, 1), points_range(3, 2),coor/2,1)]; %z6�NZ�b�T�w
%% �s�@EXCEL--------2
coordinate = [px py pz]; % +���V�ѨD�o�H�����ת��U�y��+�H���y����T�b����+�H���T�b������|��
csvFile = 'output_point.csv';
csvwrite(csvFile, coordinate);
% dos(['start ' csvFile]);

plot(px, py, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b')
hold on
% plot3(b_px, b_py, b_pz, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r')
% plot3(xHeel, yHeel, zHeel, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y')
xlabel('x�y��')
ylabel('y�y��')
% axis([100 800 -200 400 -95 105])
title('�[�u�I���G��')
grid on

