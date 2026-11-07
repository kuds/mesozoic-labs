"""Loopback-only operator interface. Forward this port with SSH for remote use."""

from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = b"""<!doctype html><html lang="en"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mesozoic Labs | Follow control</title><style>
:root{color-scheme:dark}body{margin:0;background:#142323;color:#f1f4ed;font:16px system-ui,sans-serif}
main{max-width:900px;margin:auto;padding:4vw}header,footer{display:flex;align-items:center;gap:1em;flex-wrap:wrap}
header{justify-content:space-between}h1{font-weight:550;font-size:1.5em}canvas{display:block;width:100%;background:#202e2d}
button{font:inherit;padding:.65em 1.2em;border:1px solid #b7cec0;color:inherit;background:#29423a;border-radius:4px}
#stop{background:#773d30}footer{margin-top:1em}select{font:inherit;padding:.55em}output{color:#c0d8c9}</style>
<main><header><h1>T. rex follow control</h1><strong id="state" role="status">Connecting</strong></header>
<canvas width="640" height="400" aria-label="Robot camera and selectable person boxes"></canvas>
<footer><label>Person <select aria-label="Select a person"><option value="">Select in view</option></select></label>
<button id="start">Start following</button><button id="stop">Stop</button><output></output></footer></main>
<script>
const q=s=>document.querySelector(s),can=q('canvas'),ctx=can.getContext('2d'),sel=q('select');let view=null;const session=crypto.randomUUID();
async function post(op,params={}){const r=await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({op,session,...params})});if(!r.ok)throw Error((await r.json()).error||'Action failed')}
function act(op,p){post(op,p).catch(e=>q('output').textContent=e.message)}
q('#start').onclick=()=>act('start');q('#stop').onclick=()=>act('stop');
sel.onchange=()=>sel.value && act('select',{id:Number(sel.value)});
can.onclick=e=>{if(!view)return;const r=can.getBoundingClientRect(),x=(e.clientX-r.left)*can.width/r.width,
y=(e.clientY-r.top)*can.height/r.height;const t=view.tracks.find(t=>x>=t.box[0]&&x<t.box[2]&&y>=t.box[1]&&y<t.box[3]);if(t)act('select',{id:t.id})};
async function poll(){try{const p=await (await fetch('/view',{cache:'no-store'})).json(),s=p.status;
q('#state').textContent=s.state.replaceAll('_',' ');q('output').textContent=s.forward_mps.toFixed(2)+' m/s  '+s.yaw_rps.toFixed(2)+' rad/s';
if(p.image){const im=new Image();im.src='data:image/jpeg;base64,'+p.image;await im.decode();
can.width=im.width;can.height=im.height;ctx.drawImage(im,0,0);for(const t of p.tracks){const [a,b,c,d]=t.box;ctx.strokeStyle=t.id===s.selected?'#fbcc60':'#85f9c1';
ctx.lineWidth=2;ctx.strokeRect(a,b,c-a,d-b);ctx.fillStyle=ctx.strokeStyle;ctx.font='bold 12px sans-serif';ctx.fillText('Person '+t.id,a+2,b+13)}view=p}
sel.replaceChildren(new Option('Select in view',''),...p.tracks.map(t=>new Option('Person '+t.id,t.id)));sel.value=s.selected||'';
q('#start').disabled=s.selected==null||s.armed;if(!document.hidden)await post('heartbeat')
}catch(e){q('#state').textContent='Disconnected';q('#start').disabled=true}setTimeout(poll,190)}
poll()</script></html>"""


class OperatorServer:
    def __init__(self, follower, host="127.0.0.1", port=8765, clock=time.monotonic):
        if host not in {"localhost", "127.0.0.1"}:
            raise ValueError("Bind the operator interface to loopback; use authenticated SSH forwarding.")
        self.follower = follower
        self._clock = clock
        self._lock = threading.Lock()
        self._camera = {"image": None, "tracks": []}
        self._owner = None
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, status, payload, kind="application/json"):
                body = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.headers.get("Host", "").split(":")[0] not in {"localhost", "127.0.0.1"}:
                    self.respond(403, {"error": "Use the local control page."})
                    return
                if self.path == "/":
                    self.respond(200, PAGE, "text/html; charset=utf-8")
                elif self.path == "/view":
                    with service._lock:
                        view = dict(service._camera)
                    self.respond(200, {**view, "status": service.follower.snapshot()})
                else:
                    self.respond(404, {"error": "Unknown resource"})

            def do_POST(self):
                if self.path != "/action":
                    self.respond(404, {"error": "Unknown action endpoint"})
                    return
                origin = self.headers.get("Origin")
                host_header = self.headers.get("Host", "")
                if (
                    self.headers.get_content_type() != "application/json"
                    or host_header.split(":")[0] not in {"localhost", "127.0.0.1"}
                    or (origin is not None and origin != "http://" + host_header)
                ):
                    self.respond(403, {"error": "Use the local control page."})
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 < length <= 512:
                        raise ValueError("Invalid action body")
                    data = json.loads(self.rfile.read(length))
                    now = service._clock()
                    if not isinstance(data, dict):
                        raise ValueError("Invalid action")
                    session = str(uuid.UUID(str(data.pop("session", ""))))
                    op = data.get("op")
                    with service._lock:
                        if op == "select" and set(data) == {"op", "id"}:
                            if service._owner != session and service.follower.snapshot()["armed"]:
                                raise ValueError("Stop the current session before taking control.")
                            service.follower.select(int(data["id"]), now)
                            service._owner = session
                            service.follower.heartbeat(now)
                        elif op == "heartbeat" and set(data) == {"op"}:
                            if service._owner == session:
                                service.follower.heartbeat(now)
                        elif op == "start" and set(data) == {"op"} and service._owner == session:
                            service.follower.start(now)
                        elif op == "stop" and set(data) == {"op"}:
                            service.follower.stop()
                            service._owner = None
                        else:
                            raise ValueError(
                                "Use your selected session to start; accepted actions are select, start, stop or heartbeat."
                            )
                    self.respond(200, service.follower.snapshot())
                except (ValueError, KeyError, TypeError) as error:
                    self.respond(400, {"error": str(error)})

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def publish(self, rgb):
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return
        with self._lock:
            self._camera = {"image": base64.b64encode(jpg).decode(), "tracks": self.follower.snapshot()["tracks"]}

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.follower.stop()
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join()
        self.server.server_close()
