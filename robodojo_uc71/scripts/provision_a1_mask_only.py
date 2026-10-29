#!/usr/bin/env python3
"""One explicitly authorized comparison: A0 renderer without CUDA masking."""
from pathlib import Path
import os, json, subprocess, time, signal, hashlib, datetime, argparse

BASE=Path(__file__).resolve().parents[1]
REPORT=BASE/'validation/provisioning'
RUNTIME=BASE/'runtime'
GPU_UUID='GPU-e1a460e9-e272-b227-222e-0dbb77c248c5'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt', choices=('a1_mask_only_001','a1_mask_only_002'), required=True)
    args=parser.parse_args()
    auth=json.loads((REPORT/'a1_authorization.json').read_text())
    license=json.loads((REPORT/'vendor_authorization.json').read_text())
    assert auth['diagnostics_scope_reopened'] is True and auth['full_make_toast_test_authorized'] is False
    assert license['nvidia_eula_accepted'] is True and auth['gpu_uuid']==GPU_UUID
    output=REPORT/args.attempt; output.mkdir()
    python=RUNTIME/'a0_env/bin/python'
    original=REPORT/'a0_attempt_002/provision_a0_camera.py'
    script=output/'provision_a0_camera.py'; script.write_bytes(original.read_bytes())
    assert (REPORT/'a0_camera.kit').read_bytes()==(REPORT/'a0_attempt_002/a0_camera.kit').read_bytes()
    query=['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits']
    before=subprocess.check_output(query,text=True);(output/'gpu_before.csv').write_text(before)
    rows=[line.split(',') for line in before.splitlines()]
    assert any(int(r[0])==4 and r[1].strip()==GPU_UUID and int(r[2])<256 for r in rows)
    baseline={int(r[0]):int(r[2]) for r in rows}
    assert all(memory<128 for memory in baseline.values()), 'All GPUs must be idle for unmasked startup-context diagnostic'
    context_allowance_mib=256 if args.attempt=='a1_mask_only_001' else 768
    env={k:os.environ[k] for k in ('HOME','LANG','LC_ALL','TZ') if k in os.environ}
    env.update(PATH=str(python.parent)+':/usr/local/bin:/usr/bin:/bin',
      LD_LIBRARY_PATH=str(RUNTIME/'sysroot/usr/lib/x86_64-linux-gnu')+':/opt/nvidia-drivers/lib:/usr/lib/x86_64-linux-gnu',
      PYTHONNOUSERSITE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',
      VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json',OMNI_KIT_ACCEPT_EULA='YES',
      UC71_A0_AUTHORIZED_CHILD='1',UC71_A0_OUTPUT=str(output),
      XDG_CACHE_HOME=str(RUNTIME/'cache'),XDG_CONFIG_HOME=str(RUNTIME/'config'),XDG_DATA_HOME=str(RUNTIME/'data'),
      CUDA_CACHE_PATH=str(RUNTIME/'cache/cuda'),WARP_CACHE_PATH=str(RUNTIME/'cache/warp'))
    assert 'CUDA_VISIBLE_DEVICES' not in env
    argv=['/usr/bin/gdb','--nx','--batch','--return-child-result',
      '-ex','set pagination off','-ex','set print thread-events off','-ex','set debuginfod enabled off',
      '-ex','handle SIGPIPE nostop noprint pass','-ex','run','-ex','bt 30','-ex','info sharedlibrary',
      '--args',str(python),'-X','faulthandler','-u',str(script)]
    record={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'configuration_delta':'CUDA_VISIBLE_DEVICES removed; GDB observes native crash stack',
      'argv':argv,'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
      'full_task_started':False,'model_calls':0,'automatic_retry':False,'non_target_startup_context_allowance_mib':context_allowance_mib}
    (output/'launcher_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (output/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    started=time.monotonic(); reason=None
    with (output/'stdout_stderr.log').open('w') as logfile,(output/'gpu_samples.jsonl').open('w') as samples:
      proc=subprocess.Popen(argv,cwd=RUNTIME,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
      record['pid']=proc.pid
      while proc.poll() is None:
        gpu=subprocess.check_output(query,text=True)
        samples.write(json.dumps({'elapsed_s':time.monotonic()-started,'gpu_csv':gpu})+'\n');samples.flush()
        for row in (r.split(',') for r in gpu.splitlines()):
          if int(row[0])!=4 and int(row[2])-baseline[int(row[0])]>context_allowance_mib:
            reason='non_target_gpu_allocation_guard'
        if time.monotonic()-started>240:reason='bounded_diagnostic_timeout'
        if reason:
          os.killpg(proc.pid,signal.SIGTERM)
          try:proc.wait(timeout=10)
          except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
          break
        time.sleep(1)
      record.update(returncode=proc.returncode,elapsed_s=time.monotonic()-started,termination=reason)
    (output/'gpu_after.csv').write_text(subprocess.check_output(query,text=True))
    (output/'launch_result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return 0 if proc.returncode==0 else 1

if __name__=='__main__':raise SystemExit(main())
