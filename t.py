t = 2
steps = 5
for bptt_step in reversed(range(max(0, t-steps), t)):
    print(bptt_step)
for tau in range(1,steps+1):
    if t - tau < 0:
        break
    print(t - tau)