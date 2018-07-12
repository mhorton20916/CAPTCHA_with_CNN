from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import CaptchasDotNet

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

#---------------------------------------------------------------------
# Construct the captchas object. Replace the required parameters
# 'demo' and 'secret' with the values you receive upon
# registration at http://captchas.net.
# Optional Parameters and Defaults:
# alphabet: 'abcdefghkmnopqrstuvwxyz' (Used characters in captcha)
# We recommend alphabet without mistakable ijl.
# letters: '6' (Number of characters in captcha)
# width: '240' (image width)
# height: '80' (image height)
#---------------------------------------------------------------------
    captchas = CaptchasDotNet.CaptchasDotNet (
                            client   = 'demo',
                            secret   = 'uNuZ9ELh65ImAlll4ouM6GYRefePTvd2MLdfQZ46',
                            alphabet = 'abcdefghkmnopqrstuvwxyz',
                            letters  = 6,
                            width    = 240,
                            height   = 80
                            )

    web_page = '''
<html>
<head><title>Sample Python CAPTCHA Query</title></head>
<h1>Sample Python CAPTCHA Query</h1>
<form method="get" action="check.cgi">
<table>
  <tr>
    <td>
      <input type="hidden" name="random" value="%s" />
      Your message:</td><td><input name="message" size="60" />
    </td>
  </tr>
  <tr>
    <td>
      The CAPTCHA password:
    </td>
    <td>
      <input name="password" size="16" />
    </td>
  </tr>
  <tr>
    <td>
    </td>
    <td>
       %s <br>
       <a href="%s">Phonetic spelling (mp3)</a>
    </td>
  </tr>
  <tr>
    <td>
    </td>
    <td>
      <input type="submit" value="Submit" />
    </td>
  </tr>
</table>
</form>
</html>
''' % (captchas.random (), captchas.image (), captchas.audio_url ())

    return web_page


if __name__ == "__main__":
    env_port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=env_port, debug=True)
