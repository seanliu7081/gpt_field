def sod(s):
    h,m,x=s.split(':'); return int(h)*3600+int(m)*60+float(x)
walls={'C1':(sod('05:51:23.70'),sod('07:48:50.80')),'C2':(sod('10:14:38.15'),sod('11:03:05.51')),'C3':(sod('13:36:33.0'),sod('14:51:30.0'))}
W={k:v[1]-v[0] for k,v in walls.items()}
total=sum(W.values())
live={'C1':sod('07:08:19.27')-sod('07:00:19.43'),'C3':688.653}
# model request wall-time
req_C1_union=2191.8; req_C1_in_live=370.3
req_C2=268.5299701580079
req_C3_prep=26.705963797983713+0.065197+9.110196292051114  # att1 prep, att1 failed construct, att3 prep
req_C3_exec=578.4068603368942
codex_review={'C1':sod('07:00:19.43')-sod('06:57:59.66'),'C3':(128.21660199400503-34.43609777500387)+(310.6636293379124-25.907954455935396)}
human_wait_window=sod('06:57:59.66')-sod('06:12:49.42')
nonlive_req=(req_C1_union-req_C1_in_live)+req_C2+req_C3_prep
hung=893.3681769659743+891.9334+120.00374534004368+120.00347490399145
print('walls',{k:round(v,1) for k,v in W.items()},'total',round(total,1),round(total/3600,3),'h')
live_tot=sum(live.values())
print('live exec windows',{k:round(v,1) for k,v in live.items()},'total',round(live_tot,1),f'{live_tot/total:.1%}')
allreq=req_C1_union+req_C2+req_C3_prep+req_C3_exec
print('all model-request wall time',round(allreq,1),f'{allreq/total:.1%}','; non-completing streams',round(hung,1),f'{hung/allreq:.1%} of request time',f'{hung/total:.1%} of wall')
print('model requests outside live windows',round(nonlive_req,1),f'{nonlive_req/total:.1%}')
cr=sum(codex_review.values())
print('codex review',{k:round(v,1) for k,v in codex_review.items()},round(cr,1),f'{cr/total:.1%}')
print('waiting for user authorization window',round(human_wait_window,1),f'{human_wait_window/total:.1%}')
rest=total-live_tot-nonlive_req-cr-human_wait_window
print('remainder (protocol/implementation/tests/freeze/audit/report/gaps)',round(rest,1),f'{rest/total:.1%}')
# per campaign
c1_rest=W['C1']-live['C1']-(req_C1_union-req_C1_in_live)-codex_review['C1']-human_wait_window
c2_rest=W['C2']-req_C2
c3_rest=W['C3']-live['C3']-req_C3_prep-codex_review['C3']
for k,r in (('C1',c1_rest),('C2',c2_rest),('C3',c3_rest)):
    print(k,'remainder',round(r,1),f'{r/W[k]:.1%}')
print('C1 live share',f"{live['C1']/W['C1']:.1%}",'C1 nonlive req',f"{(req_C1_union-req_C1_in_live)/W['C1']:.1%}",'C1 human wait',f"{human_wait_window/W['C1']:.1%}",'C1 codex',f"{codex_review['C1']/W['C1']:.1%}")
print('C2 req share',f"{req_C2/W['C2']:.1%}")
print('C3 live share',f"{live['C3']/W['C3']:.1%}",'C3 prep req',f"{req_C3_prep/W['C3']:.1%}",'C3 codex',f"{codex_review['C3']/W['C3']:.1%}",'C3 exec req inside live',f"{req_C3_exec/W['C3']:.1%}")
