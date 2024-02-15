t = 5
steps = 1
for bptt_step in reversed(range(max(0, t-steps), t)):
    print(bptt_step)