
export const send = () => ({ type: 'SEND_VERIFY' });

export const complete = (error, speaker) => ({ type: 'COMPLETE_VERIFY', error, speaker });
