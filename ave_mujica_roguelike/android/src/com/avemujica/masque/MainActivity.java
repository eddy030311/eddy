package com.avemujica.masque;

import android.app.Activity;
import android.graphics.Color;
import android.hardware.input.InputManager;
import android.os.Build;
import android.os.Bundle;
import android.view.Display;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.webkit.JavascriptInterface;
import android.content.SharedPreferences;
import org.json.JSONObject;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;
import java.util.Iterator;
import java.util.regex.Pattern;

/** Full-screen landscape WebView that runs the bundled game from assets/www. */
public class MainActivity extends Activity implements InputManager.InputDeviceListener {
    private WebView web;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON | WindowManager.LayoutParams.FLAG_FULLSCREEN);
        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor("#0b080d"));
        web.setOverScrollMode(View.OVER_SCROLL_NEVER);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setMediaPlaybackRequiresUserGesture(false);
        s.setAllowFileAccess(true);
        s.setAllowFileAccessFromFileURLs(true);
        web.setWebChromeClient(new WebChromeClient());
        web.setWebViewClient(new WebViewClient() {
            @Override public void onPageFinished(WebView v, String url) { scanPads(); }
        });
        ((InputManager) getSystemService(INPUT_SERVICE)).registerInputDeviceListener(this, null);
        setContentView(web);
        web.addJavascriptInterface(new Updater(), "AMNative");
        if (state != null) web.restoreState(state);
        else web.loadUrl(startUrl());
        preferHighRefresh();
        hideBars();
    }

    private void hideBars() {
        getWindow().getDecorView().setSystemUiVisibility(
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY | View.SYSTEM_UI_FLAG_FULLSCREEN
            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) hideBars();
    }

    @Override
    protected void onPause() {
        web.evaluateJavascript("try{if(G.state==='play'&&modal.hidden&&$('actCard').hidden)showPause();if(A.ctx)A.ctx.suspend()}catch(e){}", null);
        web.onPause();
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        web.onResume();
        web.evaluateJavascript("try{if(A.ctx)A.ctx.resume()}catch(e){}", null);
        hideBars();
    }

    /** Back: skip a cutscene, pause in game, close the pause menu, go back from character select, exit from the title. */
    @Override
    public void onBackPressed() {
        web.evaluateJavascript(
            "(function(){try{if(CUT.on){endCut();return 'stay'}if(G.state==='play'){if(modal.hidden&&$('actCard').hidden)showPause();else if(G.modalKind==='pause')closeModal();return 'stay'}"
            + "if(G.state==='select'){toTitle();return 'stay'}if(G.state==='title')return 'exit'}catch(e){}return 'stay'})()",
            new ValueCallback<String>() {
                @Override public void onReceiveValue(String v) {
                    if (v != null && v.contains("exit")) finish();
                }
            });
    }

    private static boolean isPadKey(int code) {
        return KeyEvent.isGamepadButton(code) || code == KeyEvent.KEYCODE_DPAD_UP || code == KeyEvent.KEYCODE_DPAD_DOWN
            || code == KeyEvent.KEYCODE_DPAD_LEFT || code == KeyEvent.KEYCODE_DPAD_RIGHT || code == KeyEvent.KEYCODE_DPAD_CENTER;
    }

    /** Forward gamepad buttons to the game (window.padKey); everything else keeps its normal path. */
    @Override
    public boolean dispatchKeyEvent(KeyEvent e) {
        int src = e.getSource();
        boolean fromPad = (src & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD || (src & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK;
        if (fromPad && e.getKeyCode() == KeyEvent.KEYCODE_BACK) {
            if (e.getRepeatCount() == 0) web.evaluateJavascript("window.padKey&&padKey(97," + (e.getAction() == KeyEvent.ACTION_DOWN) + ")", null);
            return true;
        }
        if (isPadKey(e.getKeyCode()) && (fromPad || KeyEvent.isGamepadButton(e.getKeyCode()))) {
            if (e.getRepeatCount() == 0 && (e.getAction() == KeyEvent.ACTION_DOWN || e.getAction() == KeyEvent.ACTION_UP))
                web.evaluateJavascript("window.padKey&&padKey(" + e.getKeyCode() + "," + (e.getAction() == KeyEvent.ACTION_DOWN) + ")", null);
            return true;
        }
        return super.dispatchKeyEvent(e);
    }

    private float lx, ly, lhx, lhy, lrx, lry, llt, lrt;

    private static float axis(MotionEvent e, int a, int b) {
        float v = e.getAxisValue(a);
        return Math.abs(v) > 0.01f ? v : e.getAxisValue(b);
    }

    /** Sticks, d-pad hat and analog triggers go to the game before the WebView can consume them. */
    @Override
    public boolean dispatchGenericMotionEvent(MotionEvent e) {
        if ((e.getSource() & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK && e.getAction() == MotionEvent.ACTION_MOVE) {
            float x = e.getAxisValue(MotionEvent.AXIS_X), y = e.getAxisValue(MotionEvent.AXIS_Y);
            float hx = e.getAxisValue(MotionEvent.AXIS_HAT_X), hy = e.getAxisValue(MotionEvent.AXIS_HAT_Y);
            float rx = axis(e, MotionEvent.AXIS_Z, MotionEvent.AXIS_RX), ry = axis(e, MotionEvent.AXIS_RZ, MotionEvent.AXIS_RY);
            float lt = axis(e, MotionEvent.AXIS_LTRIGGER, MotionEvent.AXIS_BRAKE), rt = axis(e, MotionEvent.AXIS_RTRIGGER, MotionEvent.AXIS_GAS);
            if (Math.abs(x - lx) > 0.02f || Math.abs(y - ly) > 0.02f || hx != lhx || hy != lhy || Math.abs(rx - lrx) > 0.02f || Math.abs(ry - lry) > 0.02f
                || Math.abs(lt - llt) > 0.05f || Math.abs(rt - lrt) > 0.05f) {
                lx = x; ly = y; lhx = hx; lhy = hy; lrx = rx; lry = ry; llt = lt; lrt = rt;
                web.evaluateJavascript("window.padAxes&&padAxes(" + x + "," + y + "," + hx + "," + hy + "," + rx + "," + ry + "," + lt + "," + rt + ")", null);
            }
            return true;
        }
        return super.dispatchGenericMotionEvent(e);
    }

    private static boolean isPad(InputDevice d) {
        if (d == null || d.isVirtual()) return false;
        int s = d.getSources();
        return (s & InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD || (s & InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK;
    }

    /** Tell the game a controller is present as soon as it is connected, not on the first button press. */
    private void scanPads() {
        for (int id : InputDevice.getDeviceIds()) {
            InputDevice d = InputDevice.getDevice(id);
            if (isPad(d)) { notifyPad(d, true); return; }
        }
    }

    private void notifyPad(InputDevice d, boolean on) {
        String name = d == null ? "" : d.getName().replace("\\", "").replace("'", "");
        web.evaluateJavascript(on ? "window.padConnected&&padConnected('" + name + "')" : "window.padDisconnected&&padDisconnected()", null);
    }

    @Override public void onInputDeviceAdded(int id) { InputDevice d = InputDevice.getDevice(id); if (isPad(d)) notifyPad(d, true); }
    @Override public void onInputDeviceChanged(int id) { }
    @Override public void onInputDeviceRemoved(int id) {
        for (int i : InputDevice.getDeviceIds()) if (isPad(InputDevice.getDevice(i))) return;
        notifyPad(null, false);
    }

    /* ------------------------------------------------------------------ online update
       The game (window.AMNative) asks for an update; the manifest on the release branch lists every file with its sha256.
       Unchanged files are copied from the current bundle, changed ones are downloaded and verified, then the new folder
       replaces the old one and the WebView reloads. Saves live in localStorage (shared by every file:// page), so they stay. */
    private static final String[] HOSTS = {
        "https://raw.githubusercontent.com/eddy030311/eddy/release/ave_mujica_roguelike/update/",
        "https://cdn.jsdelivr.net/gh/eddy030311/eddy@release/ave_mujica_roguelike/update/" };
    private static final Pattern SAFE = Pattern.compile("^[A-Za-z0-9_\\-][A-Za-z0-9_.\\-]*(/[A-Za-z0-9_\\-][A-Za-z0-9_.\\-]*)*$");
    private volatile boolean updating = false;

    private SharedPreferences prefs() { return getSharedPreferences("upd", MODE_PRIVATE); }

    private static String readAll(InputStream in) throws IOException {
        ByteArrayOutputStream o = new ByteArrayOutputStream(); byte[] b = new byte[16384]; int n;
        while ((n = in.read(b)) > 0) o.write(b, 0, n); in.close(); return o.toString("UTF-8");
    }

    private JSONObject bundledManifest() {
        try { return new JSONObject(readAll(getAssets().open("www/manifest.json"))); } catch (Exception e) { return null; }
    }

    /** the downloaded bundle, if it is newer than the one inside the APK */
    private File liveDir() {
        String d = prefs().getString("dir", null); int v = prefs().getInt("v", 0);
        JSONObject b = bundledManifest(); int bv = b == null ? 0 : b.optInt("v", 0);
        if (d == null || v <= bv) return null;
        File f = new File(d);
        return new File(f, "index.html").isFile() && new File(f, "manifest.json").isFile() ? f : null;
    }

    private String startUrl() {
        File d = liveDir();
        return d != null ? "file://" + new File(d, "index.html").getAbsolutePath() : "file:///android_asset/www/index.html";
    }

    private JSONObject currentManifest() {
        File d = liveDir();
        if (d == null) return bundledManifest();
        try { return new JSONObject(readAll(new FileInputStream(new File(d, "manifest.json")))); } catch (Exception e) { return bundledManifest(); }
    }

    private InputStream openCurrent(String path) throws IOException {
        File d = liveDir();
        return d != null ? new FileInputStream(new File(d, path)) : getAssets().open("www/" + path);
    }

    private static byte[] fetch(String url) throws IOException {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(15000); c.setReadTimeout(30000); c.setUseCaches(false);
        try {
            if (c.getResponseCode() != 200) throw new IOException("HTTP " + c.getResponseCode());
            InputStream in = c.getInputStream(); ByteArrayOutputStream o = new ByteArrayOutputStream(); byte[] b = new byte[32768]; int n;
            while ((n = in.read(b)) > 0) o.write(b, 0, n); in.close(); return o.toByteArray();
        } finally { c.disconnect(); }
    }

    private static String sha(byte[] d) throws Exception {
        byte[] h = MessageDigest.getInstance("SHA-256").digest(d); StringBuilder s = new StringBuilder();
        for (byte x : h) s.append(String.format("%02x", x)); return s.toString();
    }

    private static byte[] readBytes(InputStream in) throws IOException {
        ByteArrayOutputStream o = new ByteArrayOutputStream(); byte[] b = new byte[32768]; int n;
        while ((n = in.read(b)) > 0) o.write(b, 0, n); in.close(); return o.toByteArray();
    }

    private static void wipe(File f) {
        if (f.isDirectory()) { File[] k = f.listFiles(); if (k != null) for (File c : k) wipe(c); }
        f.delete();
    }

    private void js(final String code) { runOnUiThread(new Runnable() { public void run() { web.evaluateJavascript(code, null); } }); }

    private void doUpdate(int hi) {
        String host = HOSTS[Math.max(0, Math.min(HOSTS.length - 1, hi))];
        File tmp = new File(getFilesDir(), "www_tmp");
        try {
            JSONObject m = new JSONObject(new String(fetch(host + "manifest.json?t=" + System.currentTimeMillis()), "UTF-8"));
            JSONObject cur = currentManifest(); int v = m.getInt("v");
            if (cur != null && v <= cur.optInt("v", 0)) throw new IOException("已经是最新版本");
            JSONObject files = m.getJSONObject("files"), have = cur == null ? new JSONObject() : cur.optJSONObject("files");
            if (have == null) have = new JSONObject();
            wipe(tmp); tmp.mkdirs();
            int total = files.length(), done = 0;
            for (Iterator<String> it = files.keys(); it.hasNext(); ) {
                String p = it.next();
                if (!SAFE.matcher(p).matches() || p.contains("..")) throw new IOException("bad path " + p);
                String want = files.getJSONObject(p).getString("sha256"); byte[] data = null;
                JSONObject h = have.optJSONObject(p);
                if (h != null && want.equals(h.optString("sha256"))) { try { data = readBytes(openCurrent(p)); if (!want.equals(sha(data))) data = null; } catch (Exception e) { data = null; } }
                if (data == null) { data = fetch(host + "www/" + p); if (!want.equals(sha(data))) throw new IOException("校验失败 " + p); }
                File out = new File(tmp, p); out.getParentFile().mkdirs();
                FileOutputStream fo = new FileOutputStream(out); fo.write(data); fo.close();
                done++; if (done % 4 == 0 || done == total) js("window.updProg&&updProg(" + done + "," + total + ")");
            }
            FileOutputStream fo = new FileOutputStream(new File(tmp, "manifest.json")); fo.write(m.toString().getBytes("UTF-8")); fo.close();
            File dst = new File(getFilesDir(), "www_" + v); wipe(dst);
            if (!tmp.renameTo(dst)) throw new IOException("无法写入更新");
            String old = prefs().getString("dir", null);
            prefs().edit().putString("dir", dst.getAbsolutePath()).putInt("v", v).apply();
            if (old != null && !old.equals(dst.getAbsolutePath())) wipe(new File(old));
            js("window.updDone&&updDone(true)");
            runOnUiThread(new Runnable() { public void run() { web.postDelayed(new Runnable() { public void run() { web.loadUrl(startUrl()); } }, 900); } });
        } catch (Exception e) {
            wipe(tmp);
            String msg = String.valueOf(e.getMessage()).replace("\\", "").replace("'", "");
            js("window.updDone&&updDone(false,'" + msg + "')");
        } finally { updating = false; }
    }

    private class Updater {
        @JavascriptInterface public String platform() { return "android"; }
        @JavascriptInterface public String files() { JSONObject c = currentManifest(); JSONObject f = c == null ? null : c.optJSONObject("files"); return f == null ? "{}" : f.toString(); }
        @JavascriptInterface public void update(final int host) {
            if (updating) return; updating = true;
            new Thread(new Runnable() { public void run() { doUpdate(host); } }).start();
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        web.saveState(out);
    }

    @Override
    protected void onDestroy() {
        ((InputManager) getSystemService(INPUT_SERVICE)).unregisterInputDeviceListener(this);
        web.destroy();
        super.onDestroy();
    }

    /** Ask for the display's fastest mode at the current resolution, so the game's 120 FPS option can take effect. */
    private void preferHighRefresh() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return;
        try {
            Display d = getWindowManager().getDefaultDisplay();
            Display.Mode cur = d.getMode(), best = cur;
            for (Display.Mode m : d.getSupportedModes())
                if (m.getPhysicalWidth() == cur.getPhysicalWidth() && m.getPhysicalHeight() == cur.getPhysicalHeight() && m.getRefreshRate() > best.getRefreshRate()) best = m;
            WindowManager.LayoutParams lp = getWindow().getAttributes();
            lp.preferredDisplayModeId = best.getModeId();
            getWindow().setAttributes(lp);
        } catch (Exception ignored) { }
    }
}
