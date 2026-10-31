import React from 'react';
import { useBranding } from '../branding';

export default function BrandMark(): JSX.Element {
  const branding = useBranding();
  const logoUrl = branding.logoUrl || `${process.env.PUBLIC_URL}/logo192.png`;
  return <img src={logoUrl} alt="" className="brand-mark-image" />;
}
