clc
clear
close all
%% initial 隨機random 座標-------2
coor = 100 ;
points_range = [150, 750; -300, 300; 0, 0];

px = random('unif',points_range(1, 1), sum(points_range(1, :))/2 ,coor/2,1);           %z4為X軸變化量
py = random('unif',points_range(2, 1), points_range(2, 2),coor/2,1);        %z5為Y軸變化量
pz = random('unif',points_range(3, 1), points_range(3, 2),coor/2,1); %z6將Z軸固定

px = [px; random('unif',sum(points_range(1, :))/2 + 1, points_range(1, 2),coor/2,1)];           %z4為X軸變化量
py = [py; random('unif',points_range(2, 1), points_range(2, 2),coor/2,1)];        %z5為Y軸變化量
pz = [pz; random('unif',points_range(3, 1), points_range(3, 2),coor/2,1)]; %z6將Z軸固定
%% 製作EXCEL--------2
coordinate = [px py pz]; % +順向解求得隨機角度的各座標+隨機座標轉三軸角度+隨機三軸角度轉徑度
csvFile = 'output_point.csv';
csvwrite(csvFile, coordinate);
% dos(['start ' csvFile]);

plot(px, py, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b')
hold on
% plot3(b_px, b_py, b_pz, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r')
% plot3(xHeel, yHeel, zHeel, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y')
xlabel('x座標')
ylabel('y座標')
% axis([100 800 -200 400 -95 105])
title('加工點分佈圖')
grid on

