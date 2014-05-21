function Display120
X = [87,48,59,0,16,0,65,0,28,17,13,0,0,0,14,11,2,0,2,0,55,51,27,76,45,58,57,57,53,46,119,112,92,119,69,90,119,100,93,64,58,58,58,57,57,57,57,57,58,57,58,57,57,17,57,58,57,61,57,57,28,30,30,58,30,30,16,30,18,57]
Y = [44,14,38,0,33,22,22,35,18,52,88,48,74,59,68,53,80,80,65,69,95,103,104,85,66,90,86,87,90,92,0,119,0,78,119,73,47,12,98,100,86,90,90,86,86,86,86,86,89,87,90,88,90,54,87,90,86,96,87,87,81,81,81,89,81,81,54,81,52,87]
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
pointX0 = [0.0,48.0,65.0,87.0,17.0,0.0]
pointY0 = [0.0,14.0,22.0,44.0,52.0,35.0]
pointX1 = [0.0,11.0,14.0,13.0,0.0]
pointY1 = [48.0,53.0,68.0,88.0,80.0]
pointX2 = [45.0,76.0,51.0,27.0]
pointY2 = [66.0,85.0,103.0,104.0]
pointX3 = [92.0,119.0,119.0,112.0,69.0,64.0]
pointY3 = [0.0,0.0,78.0,119.0,119.0,100.0]
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