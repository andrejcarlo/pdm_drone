%% Script to linearize quadrotor model given by
% ElKholy, H. T. M. N. "Dynamic modeling and control of a quadrotor using linear and nonlinear 
% approaches." American University in Cairo (2014).
%
% Islam, M., M. Okasha, and M. M. Idres. "Dynamics and control of quadcopter using linear model 
% predictive control approach." IOP Conference Series: Materials Science and Engineering. Vol. 270. 
% No. 1. IOP Publishing, 2017.

%%
syms m k_tx k_ty k_tz l g I_x I_y I_z I_r k_rx k_ry k_rz k_m k_f w_r
u = sym('u', [4 1]);

% states
syms x_ y z xd yd zd phi theta psi p q r 

%  x1 = x
%  x2 = y
%  x3 = z
%  x4 = xd
%  x5 = yd
%  x6 = zd
%  x7 = phi
%  x8 = theta
%  x9 = psi
% x10 = p
% x11 = q
% x12 = r

%% Correction 
% Source: https://aviation.stackexchange.com/questions/83993/the-relation-between-euler-angle-rate-and-body-axis-rates
% relation between euler rates and angular velocities is wrong in
% paper. Correct:
% [p q r]' = K [phi_d theta_d psi_d]'
K = [1 0 -sin(theta); 0 cos(phi) sin(phi)*cos(theta); 0 -sin(phi) cos(phi)*cos(theta)];
simplify((K^-1) * [p; q; r])

% in the papar they write omega_r; they probably meant r => replace omega_r
% with r

%% 
% g(x_, y, z, xd, yd, zd, phi, theta, psi, p, q, r) = [...
%     xd;...
%     yd;...
%     zd;...
%     -m^-1 * (k_tx * xd + u(1) * (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta) ));...
%     -m^-1 * (k_ty * yd + u(1) * (sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta) ));...
%     -m^-1 * (k_tz * zd - m*g + u(1) * cos(phi)*cos(theta));...
%     p + r * cos(phi) * tan(theta) + q*sin(phi)*tan(theta);...
%     q*cos(phi) - r*sin(phi);...
%     r * cos(phi)/cos(theta) + q * sin(phi)/cos(theta);...
%     -I_x^-1 * (k_rx*p - l*u(2) - I_y*q*r + I_z*q*r + I_r*q*r);...
%     -I_y^-1 * (-k_ry*q + l*u(3) - I_x*p*r + I_z*p*r + I_r*p*r);...
%     -I_z^-1 * (u(4) - k_rz*r + I_x*p*q - I_y*p*q)...
%     ];
g(x_, y, z, xd, yd, zd, phi, theta, psi, p, q, r) = [...
    xd;...
    yd;...
    zd;...
    -m^-1 * (k_tx * xd - u(1) * (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta) ));...
    m^-1 * (-k_ty * yd - u(1) * (sin(phi)*cos(psi) - cos(phi)*sin(psi)*sin(theta) ));...
    -m^-1 * (k_tz * zd + m*g - u(1) * cos(phi)*cos(theta));...
    p + r * cos(phi) * tan(theta) + q*sin(phi)*tan(theta);...
    q*cos(phi) - r*sin(phi);...
    r * cos(phi)/cos(theta) + q * sin(phi)/cos(theta);...
    -I_x^-1 * (k_rx*p + l*u(2) - I_y*q*r + I_z*q*r + I_r*q*w_r);...
    I_y^-1 * (-k_ry*q - l*u(3) - I_x*p*r + I_z*p*r + I_r*p*w_r);...
    I_z^-1 * (-l*u(4) - k_rz*r + I_x*p*q - I_y*p*q)...
    ];


% denote states in single vector called x
x = sym('x', [12 1]);
f(x) = subs(g, transpose([x_ y z xd yd zd phi theta psi p q r]), x(1:12));

%% Linearize
Ja = jacobian(f, x);
Jb = jacobian(f, u);

%% input to rotor velocity mapping
% source: Dynamic modeling and control of a Quadrotor using linear and nonlinear approaches 
% Heba talla Mohamed Nabil Elkholy. Page 39

% Omega.^2 = K_ * u
K_ = [1/(4*k_f) 0 1/(2*k_f) 1/(4*k_m);...
    1/(4*k_f) -1/(2*k_f) 0 -1/(4*k_m);...
    1/(4*k_f) 0 -1/(2*k_f) 1/(4*k_m);...
    1/(4*k_f) 1/(2*k_f) 0 -1/(4*k_m)];

%% Insert parameters
% same as in paper
% all in SI-units
I_x = 7.5e-3;
I_y = 7.5e-3;
I_z = 7.5e-3;
l = 0.23;
I_r = 6e-5;
k_f = 3.13e-5;
k_m = 7.5e-7;
m = 0.65;
g = 9.81;
k_tx = 0.1;
k_ty = k_tx;
k_tz = k_tx;
k_rx = 0.1;
k_ry = k_rx;
k_rz = k_rx;
w_r = 0;

% insert operating point in linearization
op = [0 1 0 0 0 0 0 0 0 0 0 0]';
op_u = [m*g 0 0 0]';
A_ = subs(Ja, [x; u], [op; op_u]);
B_ = subs(Jb, [x; u], [op; op_u]);

A = double(subs(A_));
B = double(subs(B_));
K = double(subs(K_));

%% to file
s = string(K_);
s = strcat(s(:,1), ", ", s(:,2), ", ", s(:,3), ", ", s(:,4));
s = strcat("[", s(1), "; ", s(2), "; ", s(3), "; ", s(4), "]");

file = fopen("mpc_matrices.txt",'W');
fprintf(file, "self.A = " + to_python(Ja) + "\n");
fprintf(file, "self.B = " + to_python(Jb) + "\n");
fprintf(file, "\n");
fprintf(file, "A = " + string(A_) + "\n");
fprintf(file, "B = " + string(B_) + "\n");
fprintf(file, "K = " + s + "\n");
fprintf(file, "I_x = " + string(I_x)  + "\n");
fprintf(file, "I_y = " + string(I_y) + "\n");
fprintf(file, "I_z = " + string(I_z) + "\n");
fprintf(file, "l = " + string(l) + "\n");
fprintf(file, "I_r = " + string(I_r) + "\n");
fprintf(file, "k_f = " + string(k_f) + "\n");
fprintf(file, "k_m = " + string(k_m) + "\n");
fprintf(file, "m = " + string(m) + "\n");
fprintf(file, "g = " + string(g) + "\n");
fprintf(file, "k_tx = " + string(k_tx) + "\n");
fprintf(file, "k_ty = " + string(k_ty) + "\n");
fprintf(file, "k_tz = " + string(k_tz) + "\n");
fprintf(file, "k_rx = " + string(k_rx) + "\n");
fprintf(file, "k_ry = " + string(k_ry) + "\n");
fprintf(file, "k_rz = " + string(k_rz) + "\n");
fprintf(file, "w_r = " + string(k_rz) + "\n");
fclose(file);

function tmp = to_python(mat)
    tmp = "np.array(["+replace(string(mat),';',"]," + newline +" [")+"])";
    tmp = replace(tmp, "cos(", "math.cos(");
    tmp = replace(tmp, "sin(", "math.sin(");
    tmp = replace(tmp, "tan(", "math.tan(");
    tmp = replace(tmp, "pi", "math.pi ");
    tmp = replace(tmp, "^", "**");
    for i = 12:-1:1
        tmp = replace(tmp, "x" + num2str(i), "x["+num2str(i-1)+"]");
    end
    for i = 4:-1:1
        tmp = replace(tmp, "u" + num2str(i), "u["+num2str(i-1)+"]");
    end
end
