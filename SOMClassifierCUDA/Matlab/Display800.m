function Display800
X = [87,117,115,83,107,78,115,78,94,100,104,71,55,45,95,72,41,62,41,42,20,66,42,0,45,23,61,36,2,28,3,15,19,0,24,21,11,4,8,32,23,23,2,37,37,37,60,22,23,22,2,36,36,22,22,2,61,31,22,23,39,24,24,22,39,24,24,24,24,39]
Y = [98,27,119,6,0,28,90,28,22,25,66,47,11,41,41,72,3,32,22,0,68,119,107,36,73,51,91,62,70,106,0,113,21,88,91,41,84,23,100,82,51,51,71,62,62,62,90,40,51,40,71,62,61,40,40,71,91,80,40,51,20,51,51,40,20,51,51,51,51,20]
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
pointX0 = [107.0,117.0,115.0,87.0,78.0,83.0]
pointY0 = [0.0,27.0,119.0,98.0,28.0,6.0]
pointX1 = [42.0,95.0,104.0,72.0,45.0,41.0,41.0]
pointY1 = [0.0,41.0,66.0,72.0,41.0,22.0,3.0]
pointX2 = [0.0,23.0,36.0,61.0,66.0,28.0,2.0]
pointY2 = [36.0,51.0,62.0,91.0,119.0,106.0,70.0]
pointX3 = [3.0,19.0,32.0,15.0,0.0]
pointY3 = [0.0,21.0,82.0,113.0,88.0]
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