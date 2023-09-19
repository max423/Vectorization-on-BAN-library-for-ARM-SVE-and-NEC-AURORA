timev=readTimes("./timev512.txt").elapsed;
time=readTimes("./time512.txt").elapsed;


time=movmean(time,200);
timev=movmean(timev,200);

iterations = 1:size(timev);


throuhgputv = movmean((timev/10e9).^-1,200);
throuhgput = movmean((time/10e9).^-1,200);

thrfun = fit(iterations',throuhgput,'poly2');
fittedthr = thrfun(iterations);

thrvfun =fit(iterations',throuhgputv,'poly2');
fittedthrv = thrvfun(iterations);



timefun = fit(iterations',time,'poly2');
fittedtime = timefun(iterations);

timevfun = fit(iterations',timev,'poly2');
fittedtimev = timevfun(iterations);

timevmean = mean(timev);
timevstd = std(timev);


timemean = mean(time);
timestd = std(time);

figure
plot(iterations,time,'color','#77AC30');
hold on
plot(iterations,timev,'color','#0072BD');
plot(iterations,fittedtimev,'color','#4DBEEE','LineWidth',3);
plot(iterations,fittedtime,'color','#EDB120','LineWidth',3);
legend("Baseline","Enhanced","Enhanced mean","Baseline mean");
xlabel("Iterations")
ylabel("Iteration time (ns)")
ylim([min(timev),5.4*1e4]);
saveas(gcf,"time_comp.png");
close


figure
plot(iterations,throuhgput,'color','#77AC30');
hold on
plot(iterations,throuhgputv,'color','#0072BD');
plot(iterations,fittedthrv,'color','#4DBEEE','LineWidth',3);
plot(iterations,fittedthr,'color','#EDB120','LineWidth',3);
legend("Baseline","Enhanced","Enhanced mean","Baseline mean");
xlabel("Iterations");
ylabel("Throughput (iter/s)");
ylim([1.8*1e5,2.3*1e5]);
saveas(gcf,"throughput_comp.png");
close