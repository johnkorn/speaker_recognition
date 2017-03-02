
import React, { Component, PropTypes } from 'react';
import { Link } from 'react-router';
import ListUsers from '../components/ListUsers';
import '../resources/layout.css';

export default class Layout extends Component {

  static propTypes = {
    users: PropTypes.array.isRequired,
    isSend: PropTypes.bool.isRequired
  };

  static defaultProps = {
    users: [],
    isSend: false
  };

  render() {
    return (
      <div>
        <ul className="menu">
          <li className="menu__item" key="reset">
            <Link to={'/reset'} className="menu__link" activeClassName="menu__link--active">
              Reset
            </Link>
          </li>
          <li className="menu__item" key="train">
            <Link to={'/train'} className="menu__link" activeClassName="menu__link--active">
              Train
            </Link>
          </li>
          <li className="menu__item" key="verify">
            <Link to={'/verify'} className="menu__link" activeClassName="menu__link--active">
              Verify
            </Link>
          </li>
        </ul>
        <br />
        List of Users:
        <br/>
        { (!this.props.isSend)
          ? (<ListUsers users={this.props.users} />)
          : (<p> Loading... </p>)
        }
        <br/>
        {this.props.children}
      </div>
    );
  }

}