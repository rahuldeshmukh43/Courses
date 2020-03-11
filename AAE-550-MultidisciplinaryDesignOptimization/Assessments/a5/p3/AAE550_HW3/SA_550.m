function [xstar,Fstar,count,accept,oob]=SA_550(FUN,bounds,X0,OPTIONS,varargin)
%SA_550  [xstar,Fstar,count,accept,oob]=SA_550(FUN,bounds,X0,OPTIONS,varargin)
%-------------------------------------------------------------------------------
%
%  PURPOSE:   SIMULATED ANNEALING FOR GLOBAL OPTIMIZATION
%
%	CALLED BY:  any driver *.m file or from command line
%
%  EXTERNAL REFERENCES:   FUN  (name of objective function *.m file)
%
%  ENVIRONMENT:   Created using Matlab 5.3 for PC
%
%  AUTHOR:  Prof. W. A. Crossley, Purdue University, School of Aero and Astro,
%           Purdue University, 1282 Grissom Hall, W. Lafayette, IN 47907-1282
%
%  DEVELOPMENT HISTORY:
%     DATE        INITIALS    DESCRIPTION
%     11/10/02    WAC         This program uses simulated annealing to find 
%                             a minimum function value.  The approach is based
%										upon the approach for SA presented in AAE 550.
%										Currently, output options(9) not working.
%
%	INPUTS:			'FUN' 		specifies the objective function.  Additional 
%										context-specific arguments can be passed to the 
%										objective function by passing them to 'SA_550' 
%										after 'OPTIONS'.
%						'X0'			user defined initial design point.
%						'bounds' 	specifies the upper and lower limits imposed upon
%										design variable.  This matrix must have as many 
%										rows as model parameters and two columns.  First 
%										column is lower bound, second column is upper
%										bound.
%						'OPTIONS' 	specifies a number of annealing options (empty 
%										matrix or zeros for defaults):
%     								OPTIONS(1) = 	initial temperature (default = 50).  
%															This parameter is highly problem 
%															dependent. Want "large" temperature
%															to assist global search, but too 
%															large T0 will result in long run 
%															times.
%     								OPTIONS(2) = 	termination criterion (default = 
%															1.0E-06).
%     								OPTIONS(3) = 	number of successive temperature 
%															reductions (default = 4).  
%     								OPTIONS(4) = 	number of cycles before step 
%															reduction (default = 20). 
%     								OPTIONS(5) =   step corrections before temperature 
%															reduction (default = 20).
%     								OPTIONS(6) = 	temperature reduction coefficient 
%															(default = 0.50).  Typically, 0.5 
%															(rapid reduction) <= rT <= 0.85 
%															(slow reduction).															
%     								OPTIONS(7) = 	variation criterion (default =  2).
%															Used to adjust step size vector.  
%															Here, all c(i) values are same.
%										OPTIONS(8) =	maximum number of function 
%															evaluations (default = 100000).
%										OPTIONS(9) = 	output information (default = 0).  
%															0 = no output; 1 = output before 
%															temperature reduction; 2 = output 
%															before step adjustment; 3 = output 
%															after each cycle.
%	OUTPUTS:		'xstar'	is the best-ever design.
%					'Fstar' 	is the objective function value of xstar.
%					'count' 	is the total number of function evaluations 
%								(candidate designs).
%					'accept' is the number of accepted candidate designs.
%					'oob' 	is the number of candidate designs out of bounds.
%
%-------------------------------------------------------------------------------

%Check argument syntax
if nargin<3
	error('Usage: [xstar,Fstar,count,accept,oob]=SA_550(FUN,bounds,X0,OPTIONS,varargin)')
end
   
if nargin<4
	OPTIONS=[];
end

if size(bounds,2)~=2
	error('Second argument must be an nx2 matrix of parameter bounds, where n is the number of parameters.');
end

%Check OPTIONS
if isempty(OPTIONS)
   T0 = 50;				% initial temperature
   epsilon = 1E-06;	% epsilon
   Neps = 4;			% number of consecutive temperature reductions for convergence
   Ns = 20;				% number of cycles before step correction
   NT = 20;				% number of step corrections before temperature reduction
   rT = 0.5;			% temperature reduction coefficient
   c = 2;				% variation criterion
   maxeval = 100000;	% maximum function evaluations
   talk = 0;			% output flag
else
   
   if OPTIONS(1)		% initial temperature
      T0 = OPTIONS(1);
   else
      T0 = 50;				
   end
   
   if OPTIONS(2)		% epsilon
      epsilon = OPTIONS(2);
   else
      epsilon = 1E-06;
   end
   
   if OPTIONS(3)  % number of consecutive temperature reductions for convergence
      Neps = OPTIONS(3);
   else
      Neps = 4;
   end
   
   if OPTIONS(4)	% number of cycles before step correction
      Ns = OPTIONS(4);
   else
      Ns = 20;
   end
   
   if OPTIONS(5)	% number of step corrections before temperature reduction
      NT = OPTIONS(5);
   else
      NT = 20;
   end

   if OPTIONS(6)	% temperature reduction coefficient
      rT = OPTIONS(6);
   else
      rT = 0.5;
   end

   if OPTIONS(7)	% variation criterion
      c = OPTIONS(7);
   else
      c = 2.0;
   end

   if OPTIONS(8)	% maximum number of function evaluations
      maxeval = OPTIONS(8);
   else
      maxeval = 100000;
   end

   if OPTIONS(9)	% output flag
      talk = OPTIONS(9);
   else
      talk = 2.0;
   end

end

% Check bounds 
if max(bounds(:,1)>bounds(:,2))
   error('All the values in the first column of bounds must be less than those in the second.');
end

% Initialize counters, etc.
xstar = X0;
Fstar = feval(FUN,xstar,varargin{:});
X = X0;
Flast = Fstar;
count = 1;
accept = 0;
oob = 0;
fntemp = 1E6;			% large value to initialize fntemp
n = size(bounds,1);	% number of design variables
m = zeros(n,1);		% move counter for each direction
v = ones(n,1);		% initial step-vector set to all ones
I = eye(n);
ncycles = 0;
nupdate = 0;
ntemp = 0;
isame = 0;
T = T0;

disp('     Simulated Annealing Information')
disp('          Fstar           Temp          count')

% Outer loop to control convergence criteria
while (count < maxeval)
   
   %	First-level inner loop through step vector updates
   while (nupdate < NT)
      
      % Second-level inner loop through cycles
      while (ncycles < Ns)
         % begin cycles through design variable directions
         for i = 1:n
            xprime = X + (2 * rand - 1)* v .* I(:,i);	% generate a candidate point
            % while xprime(i) violates bounds, generate a new point
            while ((xprime(i) <= bounds(i,1)) | (xprime(i) >= bounds(i,2)))
               oob = oob + 1;
               xprime = X + (2 * rand - 1)* v .* I(:,i);
            end
               
            % compute function value at candidate point
            F = feval(FUN,xprime,varargin{:});
            deltaF = F - Flast;
            count = count + 1;
            
            % accept point if deltaF is negative
            if (deltaF < 0)
               Flast = F;
               X = xprime;
               accept = accept + 1;
               m(i) = m(i) + 1;
               %check Metropolis criterion for acceptance of an uphill point
            else 
               prob = exp(-deltaF / T);
               pprime = rand;
               if (pprime < prob)
                  Flast = F;
                  X = xprime;
                  accept = accept + 1;
                  m(i) = m(i) + 1;
               end
            end
            % check for best-ever point
            if (F < Fstar)
               Fstar = F;
               xstar = X;
            end
         end		% end of for loop to execute cycles
         % increment number of cycles counter
         ncycles = ncycles + 1;
      end	% end of while loop to execute Ns cycles before step-size adjustment
      
      % update step vector using m(i) information
      for i = 1:n
         if (m(i) > 0.6 * Ns)
            v(i) = v(i) * (1 + c * (m(i)/Ns - 0.6)/0.4);
         elseif (m(i) < 0.4 * Ns)
            v(i) = v(i) / (1 + c * (0.4 - m(i)/Ns)/0.4);
         end
         % ensure that new step size vector element does not exceed bounds
         if ((v(i) <= bounds(i,1)) | (v(i) >= bounds(i,2)))
            v(i) = bounds(i,2) - bounds(i,1);
         end
      end
      
      % increment number of step vector updates counter, reset number of cycles and steps
      nupdate = nupdate + 1;
      ncycles = 0;
      m = zeros(n,1);
   end	% end while loop for step vector updates
   
   % output information about best ever, current T and count
   disp([sprintf('%15.6g %15.6g %15.6g',Fstar, T, count)]);

   % check convergence criterion
   if (abs(Flast - fntemp) <= epsilon)
      isame = isame + 1;
   else
      isame = 0;
   end
   % if convergence criteria are met, then exit outermost loop for max evals
   if ((isame >= Neps) & (abs(Flast - Fstar) <= epsilon))	
      break
   end
   
   fntemp = Flast;	% function value of last accepted point at previous temperature   
   % if convergence criteria is not met, continute with temperature reduction
   T = rT * T;
   ntemp = ntemp + 1;
   % reset point to best ever, update all other counters
   X = xstar;
   F = Fstar;
   nupdate = 0;
   ncycles = 0;
   m = zeros(n,1);
   
end	% close while loop for maximum number of iterations

     
               
         
               
