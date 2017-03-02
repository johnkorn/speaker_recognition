
export const send = () => ({ type: 'SEND_TRAIN' });

export const complete = (error) => ({ type: 'COMPLETE_TRAIN', error });
