
const INITIAL_STATE = {
  send: false,
  error: null,
  data: []
};

export default (state = INITIAL_STATE, action) => {
  switch(action.type) {
    case 'SEND_USERS':
      return {
        ...state,
        error: null,
        send: true
      };
    case 'COMPLETE_USERS':
      return {
        ...state,
        send: false,
        error: action.error,
        data: action.data
      };
    case 'CLEAR_USERS':
      return {
        ...state,
        data: []
      };
    case 'ADD_USER':
      return {
        ...state,
        data: state.data.concat([{
          key: action.key,
          value: action.value
        }])
      };
    case 'UPDATE_USER':
      return {
        ...state,
        data: state.data.map(_ => (_.key === action.key)
          ? {..._, value: action.value}
          : _
        )
      };
    case 'REMOVE_USER':
      return {
        ...state,
        data: state.data.filter(_ => _.key !== action.key)
      };
    default:
      return state;
  }
}