
Epla = 0; % percent of g(x) ~= f(x)
Esvm = 0;

numTimes = 0;
numruns = 1000; % number of runs
N = 100; % number of points in the training set

for s = 1:numruns
    tPLA = ones(N,1); % creating matrix of training points
    tPLA(:, 2) = (2)*rand(N, 1)-1; % randomizing x1 and x2
    tPLA(:, 3) = (2)*rand(N, 1)-1;
    target = (2)*rand(2,2)-1; % randomizing target function
    
    
    
    % evaluating Yn using the target function
    for i = 1:N
        d = ((tPLA(i,2) - target(1,1))*(target(2,2)-target(1,2))) - ...
            ((tPLA(i,3) - target(1,2)) * (target(2,1) - target(1,1)));
        if d > 0
            y(i) = -1; %#ok<SAGROW>
        else
            y(i) = +1; %#ok<SAGROW>
        end
    end
    
    quadco = zeros(N);
    for i = 1:N
        for k = 1:N
            quadco(i, k) = y(i)*y(k)*tPLA(i,:).'*tPLA(k, :);
        end
    end
    
    % creating the initial weight vector
    gpla = zeros(1,3);

    % keep track of number of misclassified points
    nmiss = N; 
    
    while nmiss > 0
        
        % evaluating the training set using current weight vector
        for k = 1:N
            d = (gpla(1,1) * tPLA(k,1)) + (gpla(1,2) * tPLA(k,2)) ...
                + (gpla(1,3) *tPLA(k,3));
            ypla(i) = sign(d);
        end
        
        % counting the current number of misclassified points
        p = 0;
        for k = 1:N
            if y(k) ~= yPLA(k)
                p = p + 1;
            end
        end
        nmiss = p;
        
        % updating w using a random misclassified point (w = w + Ynx)
        if p > 0
            k = 0;
            while k == 0
                t = randi([1 N], 1, 1);
                if y(i) ~= yPLA(i)
                    gpla(1,1) = gpla(1,1) + (y(t)*tPLA(t,1));
                    gpla(1,2) = gpla(1,2) + (y(t)*tPLA(t,2));
                    gpla(1,3) = gpla(1,3) + (y(t)*tPLA(t,3));
                    k = 1;
                end
            end
            
%             plot the current weight vector
%             scatter(trainpts(:,2),trainpts(:,3));
%             hold on;
%             plot(target(:,1),target(:,2));
%             ezplot(w(1,1) + w(1,2)*x + w(1,3)*y == 0,[-1,1,-1,1]);
%             scatter(trainpts(:,2),trainpts(:,3));
%             hold off;
        end
    end
    
    gsvm = quadprog(quadco, -1,y.',0,0,Inf);
   
    % to calculate probability of g(x) ~= f(x)
    testpts = ones(100000,1); % creating matrix of test points
    testpts(:, 2) = (2)*rand(100000, 1)-1; % randomizing x1 and x2
    testpts(:, 3) = (2)*rand(100000, 1)-1;
    wrongptspla = 0;
    wrontptssvm = 0;
    
    % evaluate the test points. 
     for i = 1:100000
         d = ((testpts(i,2) - target(1,1))*(target(2,2)-target(1,2))) - ...
             ((testpts(i,3) - target(1,2)) * (target(2,1) - target(1,1)));
         if d > 0
             testy(i) = -1;
         else
             testy(i) = +1;
         end
     end
    
    for k = 1:100000
        result = (gpla(1,1) * testpts(k,1)) + (gpla(1,2) * testpts(k,2)) ...
            + (gpla(1,3) *testpts(k,3));
        testypla(k) = sign(result);
        result2 = (gsvm(1,1) * testpts(k,1)) + (gsvm(1,2) * testpts(k,2)) ...
            + (gsvm(1,3) *testpts(k,3));
        testysvm(k) = sign(result2);
    end
    
    % count number of test points misclassified
    for k = 1:100000
        if testy(k) ~= testypla(k)
            wrongptspla = wrongptspla + 1;
        end
        if testy(k) ~= testysvm(k)
            wrongptssvm = wrongptssvm + 1;
        end
    end

    % calculating the percent of wrong points
    wrongpla = wrongptspla/100000;
    Epla = Epla + wrongpla;
    wrongsvm = wrongptssvm/100000;
    Esvm = Esvm + wrongsvm;
    
    if Esvm < Epla
        numTimes = numTimes + 1;
    end
end

% calculating the average per each run
Epla = Epla / numruns;
Esvm = Esvm / numruns;

% display result

disp('Probability of Wrong Prediction (PLA):');
disp(Epla);
disp('Probability of Wrong Prediction (SVM):');
disp(Epla);
disp('Percentage of time when SVM is better than PLA:');
disp(numTimes/numruns);
                
        