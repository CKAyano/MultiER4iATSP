clc
clear
close all

pointInner = csvread('output_point.csv');
% pointOuter = csvread('PointPlanningImportOuterCsv.csv');
% pointHeel = csvread('PointPlanningImportHeelCsv.csv');

% xHeel = pointHeel(:, 1);
% yHeel = pointHeel(:, 2); 
% zHeel = pointHeel(:, 3);

Px = pointInner(:,1);
Py = pointInner(:,2);
Pz = pointInner(:,3);

% a_px = Px([4]);
% a_py = Py([4]);
% a_pz = Pz([4]);

% b_px = Px([5, 3, 1, 2]);
% b_py = Py([5, 3, 1, 2]);
% b_pz = Pz([5, 3, 1, 2]);

% plot3(Px, Py, Pz, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b')
plot(Px, Py, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b')
hold on
% plot3(b_px, b_py, b_pz, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r')
% plot3(xHeel, yHeel, zHeel, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y')
xlabel('x座標')
ylabel('y座標')
% axis([100 800 -200 400 -95 105])
title('加工點分佈圖')
grid on