import React from 'react';
import { Box, useStyleConfig } from '@chakra-ui/react';

const Card = React.forwardRef(({ variant, children, ...rest }, ref) => {
  const styles = useStyleConfig('Card', { variant });
  return (
    <Box ref={ref} __css={styles} {...rest}>
      {children}
    </Box>
  );
});

export default Card;
