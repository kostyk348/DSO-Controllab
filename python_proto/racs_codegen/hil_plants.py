"""Plant C implementations for HIL (Euler, Ts=0.01). Matches racs2.py plants."""

PLANTS_C = {
    'SecondOrderDelay': r"""
#define TS 0.01f
#define DELAY_N 200
static float px1, px2;
static float dbuf[DELAY_N]; static int dbi;
static void plant_reset(void){ px1=0; px2=0; for(int i=0;i<DELAY_N;i++)dbuf[i]=0; dbi=0; }
static float plant_step(float u){
    float ud = dbuf[dbi]; dbuf[dbi]=u; dbi=(dbi+1)%DELAY_N;
    px1 += (-px1 + ud)/1.5f * TS;
    px2 += (-px2 + px1)/0.8f * TS;
    return px2;
}
""",
    'FourthOrder': r"""
#define TS 0.01f
static float px[4];
static void plant_reset(void){ for(int i=0;i<4;i++)px[i]=0; }
static float plant_step(float u){
    float dx0 = -4*px[0]-6*px[1]-4*px[2]-px[3]+u;
    float dx1 = px[0], dx2 = px[1], dx3 = px[2];
    px[0]+=dx0*TS; px[1]+=dx1*TS; px[2]+=dx2*TS; px[3]+=dx3*TS;
    return px[3];
}
""",
    'Underdamped': r"""
#define TS 0.01f
static float px[2];
static void plant_reset(void){ px[0]=0; px[1]=0; }
static float plant_step(float u){
    float wn2 = 2.25f;
    float dx0 = px[1];
    float dx1 = -0.45f*px[1] - wn2*px[0] + wn2*u;
    px[0]+=dx0*TS; px[1]+=dx1*TS;
    return px[0];
}
""",
    'NonMinPhase': r"""
#define TS 0.01f
static float px[3];
static void plant_reset(void){ for(int i=0;i<3;i++)px[i]=0; }
static float plant_step(float u){
    float dx0 = -3*px[0]+px[1];
    float dx1 = -3*px[1]+px[2];
    float dx2 = -px[2]+u;
    px[0]+=dx0*TS; px[1]+=dx1*TS; px[2]+=dx2*TS;
    return px[0] - 2*px[1];
}
""",
    'IntegratingDelay': r"""
#define TS 0.01f
#define DELAY_N 100
static float pstate;
static float dbuf[DELAY_N]; static int dbi;
static void plant_reset(void){ pstate=0; for(int i=0;i<DELAY_N;i++)dbuf[i]=0; dbi=0; }
static float plant_step(float u){
    float ud = dbuf[dbi]; dbuf[dbi]=u; dbi=(dbi+1)%DELAY_N;
    pstate += ud*TS;
    return pstate;
}
""",
}

PLANT_NAMES = list(PLANTS_C.keys())
