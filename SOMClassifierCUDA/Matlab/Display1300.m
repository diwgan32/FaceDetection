function Display1300
X = [36,0,22,20,0,42,38,21,8,20,30,83,54,73,0,71,50,64,102,64,114,100,112,50,99,102,105,105,104,105,86,0,65,55,54,88,68,85,38,35,105,102,102,105,101,105,101,101,101,101,102,102,102,101,101,102,105,102,101,101,101,101,101,101,101,101,101,103,101,101]
Y = [42,44,0,23,1,17,15,49,31,49,110,102,118,71,118,91,107,104,63,104,35,119,73,0,0,60,23,23,62,63,45,68,27,59,86,8,51,9,71,84,23,60,60,23,61,23,61,61,60,61,60,60,60,61,61,60,23,60,61,61,61,61,61,61,61,61,61,63,61,61]
c=zeros(70, 3)
for j=1:10,
    c(j, 1) = 0;
    c(j, 2) = 0;
    c(j, 3) = 1;
end
for j=11:20,
   c(j, 1) = 1;
  c(j, 2) = 0;
    c(j, 3) = 0;
end
for j=21:30,
    c(j, 1) = 0;
    c(j, 2) = 1;
    c(j, 3) = 0;
end
for j=31:40,
    c(j, 1) = 0;
    c(j, 2) = 0;
    c(j, 3) = 0;
end
for j=41:70,
    c(j, 1) = 1;
    c(j, 2) = 0;
    c(j, 3) = 1;
end
pointX0 = [22.0,42.0,36.0,21.0,20.0,0.0,0.0]
pointY0 = [0.0,17.0,42.0,49.0,49.0,44.0,1.0]
pointX1 = [102.0,83.0,54.0,0.0,73.0]
pointY1 = [63.0,102.0,118.0,118.0,71.0]
pointX2 = [50.0,99.0,114.0,112.0,100.0]
pointY2 = [0.0,0.0,35.0,73.0,119.0]
pointX3 = [88.0,86.0,54.0,35.0,0.0,85.0]
pointY3 = [8.0,45.0,86.0,84.0,68.0,9.0]
figure
grid on
hold on
fill( pointX0, pointY0, 'b')
alpha(0.3)
hold on
fill(pointX1, pointY1, 'r')
alpha(0.3)
hold on
fill(pointX2, pointY2, 'g')
alpha(0.3)
hold on
fill(pointX3, pointY3, 'black')
alpha(0.3)
h = scatter(X,Y, 70, c);
end
