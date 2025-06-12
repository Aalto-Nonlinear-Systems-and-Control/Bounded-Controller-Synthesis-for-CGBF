clear; echo off;
close all;
tic

pvar x y;
dpvar delta lambda;
vars = [x, y];

xi0 = 1e-8;


ds = 8;
du = 3;

monos_ux = monomials(vars,0:du);

prog = sosprogram(vars,[lambda, delta]);

[prog, u1] = sospolyvar(prog, monos_ux);
[prog, u2] = sospolyvar(prog, monos_ux);

% safe set and target set (h > 0)
h =  -(4*(x-2) - 2*y^3)^2 + 0.8*y^3 + 10;

% goal region (g < 0)
g = (((x-2) - 3.8)^2 / 1.2^2) + ((y - 1.9)^2 / 0.4^2) - 1;

Lhu = diff(h,vars(1))*u1+diff(h,vars(2))*u2;

% auxiliary polynomials
monos_s = monomials(vars,0:ds);

[prog, s0] = sospolyvar(prog, monos_s);
[prog, s1] = sospolyvar(prog, monos_s);

% #TAG constraints
prog = sosineq(prog, Lhu-lambda*h + delta - s0*h - s1*g);
prog = sosineq(prog, delta);
prog = sosineq(prog, lambda-xi0);
prog = sosineq(prog, s0);
prog = sosineq(prog, s1);

% #TAG set objective function
prog = sossetobj(prog,delta);

solver_opt.solver = 'mosek';

prog = sossolve(prog, solver_opt);

u1_poly = sosgetsol(prog,u1);
u2_poly = sosgetsol(prog,u2);
lambda_poly = sosgetsol(prog,lambda);
delta_poly = sosgetsol(prog,delta);

disp("u1_poly:----------------------------------------------------------")
disp(poly2str(u1_poly));
disp("u2_poly:----------------------------------------------------------")
disp(poly2str(u2_poly));
disp(poly2str(lambda_poly));
disp(poly2str(delta_poly));
disp(poly2str(delta_poly/lambda_poly));


% #SUPTAG poly plot

syms x y;

P = str2sym(poly2str(u1_poly));
Q = str2sym(poly2str(u2_poly));

x_min = -2;
x_max = 6;
y_min = -3;
y_max = 3;

hold on; axis equal;

plotPolyL(-g, vars, [x_min, x_max], [y_min, y_max], [0 0.4470 0.7410], [0 0]);
plotPolyL(h, vars, [x_min, x_max], [y_min, y_max], [0.8500 0.3250 0.0980], [0 0]);


% plotVectorFieldDirection(P,Q,[x_min,y_min],[x_max, y_max],70);
plotStreamlines(P,Q,[x_min, x_max],[y_min, y_max],100);


function plotStreamlines(P, Q, xrange, yrange, density, numStart)
% 绘制二维向量场的流线图
% 输入参数：
%   P      : x分量的表达式（函数句柄/字符串/符号表达式）
%   Q      : y分量的表达式
%   xrange : x轴范围，默认[-5,5]
%   yrange : y轴范围，默认[-5,5]
%   density: 网格密度，默认30
%   numStart: 起始点数量（每方向），默认15

% 处理输入参数
if nargin < 6, numStart = 15; end
if nargin < 5, density = 30; end
if nargin < 4, yrange = [-5 5]; end
if nargin < 3, xrange = [-5 5]; end

% 转换输入为函数句柄
P = expr2fun(P);
Q = expr2fun(Q);

% 生成计算网格
x = linspace(xrange(1), xrange(2), density);
y = linspace(yrange(1), yrange(2), density);
[X, Y] = meshgrid(x, y);

% 计算向量场
U = P(X, Y);
V = Q(X, Y);

% 生成起始点
startx = linspace(xrange(1)*0.9, xrange(2)*0.9, numStart);
starty = linspace(yrange(1)*0.9, yrange(2)*0.9, numStart);
[startX, startY] = meshgrid(startx, starty);

% 计算流线
% streamOptions = [0.1, 1000]; % [步长, 最大点数]
% quiver(X,Y,U,V);
% verts = streamslice(X, Y, U, V, startX(:), startY(:));
s= streamslice(X,Y,U,V,3);
set(s,'Color',[146/255 149/255 145/255])

% 绘制图形
% figure
% hold on
% for k = 1:length(verts)
%     v = verts{k};
%     if ~isempty(v)
%         plot(v(:,1), v(:,2), 'LineWidth', 1.2, 'Color', [0.1 0.4 0.7])
%     end
% end

% 图形修饰
axis([xrange yrange])
xlabel('y1'), ylabel('y2')
title('')
grid on
set(gca, 'Box', 'on')
hold off
end

% 辅助函数：将各种输入转换为函数句柄
function fh = expr2fun(expr)
if isa(expr, 'function_handle')
    fh = expr;
elseif ischar(expr)
    fh = str2func(['@(x,y)' vectorize(expr)]);
elseif isa(expr, 'sym')
    vars = symvar(expr);
    if numel(vars) < 2
        vars = [sym('x'), sym('y')]; % 强制使用x,y作为变量
    end
    fh = matlabFunction(expr, 'Vars', vars(1:2));
else
    error('不支持的输入类型');
end
end


function plotVectorFieldDirection(P, Q, xrange, yrange, density)
% 绘制二维向量场（仅显示方向）
% 修改说明：对向量场进行归一化处理，使所有箭头长度相同

% 处理P的输入类型
if ischar(P)
    P = str2func(['@(x,y)' vectorize(P)]);
elseif isa(P, 'sym')
    vars = symvar(P);
    if numel(vars) < 2
        vars = [sym('x'), sym('y')]; % 默认变量为x,y
    end
    P = matlabFunction(P, 'Vars', vars(1:2));
end

% 处理Q的输入类型
if ischar(Q)
    Q = str2func(['@(x,y)' vectorize(Q)]);
elseif isa(Q, 'sym')
    vars = symvar(Q);
    if numel(vars) < 2
        vars = [sym('x'), sym('y')]; % 默认变量为x,y
    end
    Q = matlabFunction(Q, 'Vars', vars(1:2));
end

% 生成网格
x = linspace(xrange(1), xrange(2), density);
y = linspace(yrange(1), yrange(2), density);
[X, Y] = meshgrid(x, y);

% 计算原始向量分量
U = P(X, Y);
V = Q(X, Y);

% 核心修改：归一化向量场
% magnitude = sqrt(U.^2 + V.^2);          % 计算向量长度
% non_zero = magnitude > 0;               % 找到非零向量位置
% U(non_zero) = U(non_zero) ./ magnitude(non_zero);  % 归一化x分量
% V(non_zero) = V(non_zero) ./ magnitude(non_zero);  % 归一化y分量

% 绘制参数调整
% scale = 0.7;  % 箭头显示比例（根据密度调整）
% quiver(X, Y, U, V, scale, ...         % 固定箭头长度
%        'LineWidth', 1.2, ...          % 线宽
%        'MaxHeadSize', 0.5, ...        % 箭头头部大小
%        'AutoScale', 'off', ...        % 关闭自动缩放
%        'Color', [0.2 0.4 0.8]);       % 颜色设置

stream2(X,Y,U,V,X,Y);

axis tight;
xlabel('x'); ylabel('y');
title('归一化向量场方向');
grid on;
end

function s=poly2str(p)
    c=char(p);
    s=c{1};
end


