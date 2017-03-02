import React, { Component } from 'react';
import logo from '../resources/logo.svg';
import '../resources/App.css';

export default class App extends Component {

  render() {
    return (
      <div className="App">
        <div className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h2>Speaker Verification Demo</h2>
        </div>
        <div className="App-intro">
          {this.props.children}
        </div>
      </div>
    );
  }

}
