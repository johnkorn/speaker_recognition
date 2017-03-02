
import React, { Component, PropTypes } from 'react';
import Recorder from '../api/recoder';
import '../resources/button.css';

export default class Timer extends Component {

  static propTypes = {
    long: PropTypes.number.isRequired,
    max: PropTypes.number.isRequired,
    onAdd: PropTypes.func.isRequired,
    onDelete: PropTypes.func.isRequired
  };

  state = {
    files: [],

    recording: false,
    playing: false,
    timer: '00',
  };

  constructor(props) {
    super(props);
    this.audioContext = null;
    this.recorder = null;
    this.snd = null;
  }

  componentDidMount() {

    try {
      window.AudioContext = window.AudioContext || window.webkitAudioContext;
      navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia;
      window.URL = window.URL || window.webkitURL;

      this.audioContext = new AudioContext();
      console.log('Audio context set up.');
      console.log('navigator.getUserMedia ' + (navigator.getUserMedia ? 'available.' : 'not present!'));
    } catch (e) {
      console.error('No web audio support in this browser!', e);
    }

    navigator.getUserMedia({audio: true}, (stream) => {
      const input = this.audioContext.createMediaStreamSource(stream);
      console.log('Media stream created.');

      this.recorder = new Recorder(input);
      console.log('Recorder initialised.');
    }, (e) => {
      console.error('No live audio input: ' + e);
    });
  }

  componentWillUnmount() {
    this.recorder && this.recorder.clear();
    this.audioContext = null;
    this.recorder = null;
    this.interval && clearInterval(this.interval);
    this.interval = null;
  }

  _onStart = () => {
    this.recorder && this.recorder.record();
    const time = Date.now();
    setTimeout(() => {
      if (this.state.recording === time) {
        this._onStop();
      }
    }, 10 * 1000);
    this.interval = setInterval(() => {
      const timer = parseInt((Date.now() -  this.state.recording) / 1000, 0);
      this.setState({
        recording: time,
        timer: (timer > 9) ? timer.toString() : `0${timer}`
      });
    }, 500);
    console.log('Recording...');
    this.setState({
      recording: time,
      timer: '00'
    });
  };

  _onStop = () => {
    if (this.recorder) {
      this.recorder.stop();
      console.log('Stopped recording.');
      if (parseInt(this.state.timer, 0) >= this.props.long) {
        try {
          this.recorder.exportWAV((blob) => {
            const filename = new Date().toISOString();
            this.setState({
              files: this.state.files.concat([{
                key: filename,
                data: blob
              }])
            });
            this.props.onAdd(filename, blob);
          });
        } catch (e) {
          console.error(e);
        }
      }
      this.recorder.clear();
    }
    this.interval && clearInterval(this.interval);
    this.interval = null;
    this.setState({
      recording: null,
      timer: '00'
    });
  };

  _onPlay = (blob) => {
    this.snd && this.snd.pause();
    this.snd = new Audio(URL.createObjectURL(blob));
    this.snd.play();
    this.snd.addEventListener("ended", () => {
      this.snd.currentTime = 0;
      this.setState({
        playing: false
      });
    });
    this.setState({
      playing: true
    });
  };

  _onPause = () => {
    this.snd && this.snd.pause();
    this.setState({
      playing: false
    });
  };

  _onDelete = (id) => {
    this.snd && this.snd.pause();
    this.setState({
      files: this.state.files.filter(_ => _.key !== id),
      playing: false
    });
    this.props.onDelete(id);
  };

  render() {
    return (
      <div>
        <p> Input files: </p>
        <ul>
          {this.state.files.map(file => (
            <li key={file.key}>
              <p>
                {file.key}
              </p>
              <p style={{display: (!this.state.recording) ? 'initial' : 'none'}}>
                <input type="button" value="Play Record" onClick={(e) => this._onPlay(file.data)} /> &nbsp;
                <input type="button" value="Stop Record" onClick={(e) => this._onPause()} /> &nbsp;
                <input type="button" value="Delete Record" onClick={(e) => this._onDelete(file.key)} />
              </p>
            </li>
          ))}
        </ul>
        <p style={{display: (!this.state.playing && this.state.files.length < this.props.max) ? 'initial' : 'none'}}>
          <input
            type="button"
            value="Start"
            className="btn-controll"
            disabled={this.state.recording}
            onClick={() => this._onStart()}
          />
          &nbsp;
          <span style={{
            color: (parseInt(this.state.timer, 0) >= this.props.long)
              ? 'green'
              : (
                (parseInt(this.state.timer, 0) === 0)
                  ? 'black' : 'red'
              )
          }}>
            {this.state.timer}
          </span>
          &nbsp;
          <input
            type="button"
            value="Stop"
            className="btn-controll"
            disabled={!this.state.recording}
            onClick={() => this._onStop()}
          />
        </p>
      </div>
    );
  }

}