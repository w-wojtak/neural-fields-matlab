function status = TimeOutput(t,u,flag,p,x)

  if isempty(flag)
    disp(['t = ' num2str(max(t))]);
  end

  if ~isempty(u)
    if size(u,2) == 0
      PlotSpot1DTimeStep(u,p,x,t);
    else
      PlotSpot1DTimeStep(u(:,end),p,x,t);
    end
  end

  status = 0;

end

