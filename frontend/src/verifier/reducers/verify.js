
const INITIAL_STATE = {
  send: false,
  speaker: null
};

export default (state = INITIAL_STATE, action) => {
  switch (action.type) {
    case 'SEND_VERIFY':
      return {
        ...state,
        send: true,
        speaker: null
      };
    case 'COMPLETE_VERIFY':
      return {
        ...state,
        send: false,
        speaker: action.speaker
      };
    default:
      return state;
  }
}