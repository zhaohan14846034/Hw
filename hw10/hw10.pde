// Simple Harmonic Motion (Spring-Mass) in Processing
// m x'' + c x' + k x = 0
// Controls:
//   LEFT/RIGHT : decrease/increase k
//   UP/DOWN    : set initial displacement (amplitude) bigger/smaller (reset)
//   D          : toggle damping
//   P          : pause/resume
//   R          : reset
//   Mouse drag : set mass position (and reset velocity)

float m = 1.0;      // mass
float k = 12.0;     // spring constant
float c = 0.0;      // damping (0 => pure SHM)
boolean dampingOn = false;

float x = 1.5;      // displacement from equilibrium (physics units)
float v = 0.0;      // velocity
float a = 0.0;      // acceleration

float eqX;          // equilibrium x in pixels
float centerY;
float scale = 120;  // pixels per physics unit

boolean paused = false;
float lastT;

void setup() {
  size(900, 420);
  eqX = width * 0.55;
  centerY = height * 0.55;
  lastT = millis() / 1000.0;
  smooth();
}

void resetSystem(float x0) {
  x = x0;
  v = 0;
}

void draw() {
  background(250);

  // time step (seconds)
  float t = millis() / 1000.0;
  float dt = t - lastT;
  lastT = t;
  dt = constrain(dt, 0, 0.033); // avoid huge jumps (tab-switch etc.)

  if (!paused) {
    // physics: a = -(k/m) x - (c/m) v
    a = -(k/m) * x - (c/m) * v;

    // semi-implicit Euler (more stable than naive Euler)
    v += a * dt;
    x += v * dt;
  }

  // ----- draw spring + mass -----
  float massX = eqX + x * scale;

  // guide line (equilibrium)
  stroke(180);
  line(eqX, 40, eqX, height - 40);

  // wall
  stroke(60);
  strokeWeight(4);
  line(60, centerY - 70, 60, centerY + 70);

  // spring (zig-zag)
  strokeWeight(2);
  stroke(30);
  drawSpring(60, centerY, massX - 25, centerY, 16, 14);

  // mass
  noStroke();
  fill(30, 140, 255);
  ellipse(massX, centerY, 50, 50);

  // ----- UI text -----
  fill(20);
  textSize(14);
  text("SHM: m x'' + c x' + k x = 0", 18, 22);
  text(String.format("m=%.2f  k=%.2f  c=%.2f  (damping %s)", m, k, c, dampingOn ? "ON" : "OFF"), 18, 42);
  text(String.format("x=%.3f  v=%.3f  a=%.3f", x, v, a), 18, 62);
  text("Controls: LEFT/RIGHT k | UP/DOWN amplitude(reset) | D damping | P pause | R reset | Drag mouse", 18, 82);

  // small plot (x vs time trail)
  drawMiniTrail(x);
}

void drawSpring(float x1, float y1, float x2, float y2, int coils, float amp) {
  // horizontal spring only (y1==y2)
  float L = x2 - x1;
  float step = L / (coils * 2.0);
  float px = x1;
  float py = y1;

  beginShape();
  vertex(px, py);
  for (int i = 1; i < coils * 2; i++) {
    px = x1 + i * step;
    py = y1 + ((i % 2 == 0) ? -amp : amp);
    vertex(px, py);
  }
  vertex(x2, y2);
  endShape();
}

// ----- mini trail plot -----
float[] trail = new float[260];
int trailIdx = 0;

void drawMiniTrail(float xval) {
  // store
  trail[trailIdx] = xval;
  trailIdx = (trailIdx + 1) % trail.length;

  // plot area
  int px = 18, py = 110, pw = 360, ph = 120;
  stroke(210);
  noFill();
  rect(px, py, pw, ph);

  // center line
  stroke(220);
  line(px, py + ph/2, px + pw, py + ph/2);

  // plot
  stroke(255, 80, 80);
  noFill();
  beginShape();
  for (int i = 0; i < trail.length; i++) {
    int idx = (trailIdx + i) % trail.length;
    float xv = map(i, 0, trail.length - 1, px, px + pw);
    float yv = map(trail[idx], -2.5, 2.5, py + ph, py);
    vertex(xv, yv);
  }
  endShape();

  fill(60);
  text("x(t) trail", px + 6, py + 16);
}

void keyPressed() {
  if (keyCode == LEFT)  k = max(0.5, k - 0.8);
  if (keyCode == RIGHT) k = min(80, k + 0.8);

  if (keyCode == UP)    resetSystem(min(3.0, x + 0.3));
  if (keyCode == DOWN)  resetSystem(max(-3.0, x - 0.3));

  if (key == 'r' || key == 'R') resetSystem(1.5);

  if (key == 'p' || key == 'P') paused = !paused;

  if (key == 'd' || key == 'D') {
    dampingOn = !dampingOn;
    c = dampingOn ? 1.2 : 0.0; // toggle damping strength
  }
}

void mouseDragged() {
  // set displacement by dragging mass
  float massX = mouseX;
  x = (massX - eqX) / scale;
  v = 0;
}
