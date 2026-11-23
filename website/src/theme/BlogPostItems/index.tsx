import React from 'react';
import {useLocation} from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';
import BlogPostItems from '@theme-original/BlogPostItems';
import type {Props} from '@theme/BlogPostItems';

export default function BlogPostItemsWithHeading(props: Props) {
  const {pathname} = useLocation();
  const blogIndex = useBaseUrl('/blog');
  return (
    <>
      {pathname.replace(/\/$/, '') === blogIndex && <h1>Blog</h1>}
      <BlogPostItems {...props} />
    </>
  );
}
