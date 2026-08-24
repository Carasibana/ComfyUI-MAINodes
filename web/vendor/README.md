# Vendored muxers (third party, MIT)

The precise export path in `../video_compare.js` encodes frames with
WebCodecs and needs a container writer in the browser. Two are vendored
here as UMD globals (`Mp4Muxer`, `WebMMuxer`); the widget injects them with
a `<script>` tag from this extension's own path, so nothing is fetched from
a CDN at runtime.

| file | upstream | version | license |
|---|---|---|---|
| `mp4-muxer.min.js` | https://github.com/Vanilagy/mp4-muxer (npm `mp4-muxer`) | 5.2.1 | MIT |
| `webm-muxer.min.js` | https://github.com/Vanilagy/webm-muxer (npm `webm-muxer`) | 5.1.2 | MIT |

Both files are the jsDelivr minified builds and keep their original header
comment naming the upstream file and version. They are unmodified: refresh
them by re-downloading the same npm build, never by hand editing. Their MIT
license is the upstream project's and is separate from this pack's license.
